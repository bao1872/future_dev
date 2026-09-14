"""PGM-BAR-0 — 5m dynamic path + SMC transition increment over episode endpoint

===========================================================================
唯一研究问题（严格可证伪的架构升级实验）
===========================================================================
在已证明有效的 15-mask pairwise CRF endpoint head（PGM-0）之上，把预测从
"episode 起点状态 → endpoint mask" 升级为 "动态 5m hazard + endpoint CRF"：

    P(H_{t+1}=1, E_{t+1} | F_t) = P(H_{t+1}=1 | F_t) · P(E_{t+1} | H=1, F_t)

每根 5m bar t 收盘后，用仅依赖 <= t 的信息预测：
    * 下一根 bar 是否结束当前 structural episode（hazard）
    * 若结束，以哪个 4-bit endpoint mask 结束（CRF head）

五个严格嵌套模型回答四个增量问题：

    M0  START          = 4 start-geometry + 10 functional provenance
                         + prev_event_mask + elapsed_log
    M1  BAR            = M0 + 当前 5m snapshot（dU/dD/width/log_ratio @ t）
    M2  PATH           = M1 + 6 个 causal running-path 特征
    M3  SMC            = M2 + 8 BOS/CHoCH bits + internal/swing bias
    M4  OB             = M3 + 7 Order-Block lifecycle 特征

    M1-M0  = 5m 动态更新价值（知道当前位置 vs 只知道起点）
    M2-M1  = 当前位置已知后，走过来的路径是否还有价值
    M3-M2  = BOS/CHoCH 结构是否有独立增量
    M4-M3  = OB lifecycle 是否有独立增量

主指标 = mean episode NLL（整个 episode 活多久 + 如何结束），不是 per-bar accuracy。

===========================================================================
冻结边界（GOVERNANCE）
===========================================================================
* SMC 直接使用 repo 现有 canonical `panji_indicators.compute_smc_pine`，
  禁止自行重写 BOS/CHoCH/OB；第一轮严格遵守 canonical default 参数（不 tuning）。
  OB 第一轮只使用 canonical 默认启用的 internal OB（swing OB 默认关闭）。
* SMC 计算输入在进入 compute_smc_pine 前已物理截断到 TB3：
  tb4_analytically_used = false。loader 可为 block grid 读取完整 raw 文件，
  但所有数值数组使用前已截断为 <= TB3 view；本脚本不使用任何 TB4 数值。
* 不重跑 / 不修改 EPISODE-0、LOCAL-0、PGM-0、STATE/SEQ 等历史代码与输出。
* 禁止：RL / PnL / latent state / HMM / HSMM / 新 Sweep 定义 / SMC 参数 tuning /
  Path Signature / 三阶 endpoint interaction / 修改 episode 定义 /
  修改 canonical panji_indicators.py。

===========================================================================
输出
===========================================================================
    pgm_bar0_summary.json
    pgm_bar0_sample_audit.json
    pgm_bar0_model_metrics.csv
    pgm_bar0_bootstrap.csv
    pgm_bar0_by_symbol.csv
    pgm_bar0_smc_event_counts.csv
    pgm_bar0_causality_audit.csv
    pgm_bar0_optimizer_audit.csv
（大型 pgm_bar0_samples.parquet 存 gitignored cache）
"""
from __future__ import annotations

import argparse
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

# ---- reused canonical components (READ-ONLY) ---------------------------------
from research.liquidity_oracle_atlas.experiment_pgm0_endpoint_coupling_v1 import (  # noqa: E402
    fit_conditional_crf, predict_mask_prob, mask_index, densify,
)
from research.liquidity_oracle_atlas.experiment_local_liquidity_transition_v0 import (  # noqa: E402
    FULL_UNIV, TRAIN_BLOCK, TEST_BLOCK, OUT, CACHE, build_blocks,
)
from research.liquidity_oracle_atlas.run_5m_graph_probability_v1 import (  # noqa: E402
    load_raw_bars,
)
from research.liquidity_oracle_atlas.experiment_step01_cause_decomposition_v1 import (  # noqa: E402
    load_seq,
)
from research.liquidity_oracle_atlas.experiment_episode0_dynamic_episodes_v1 import (  # noqa: E402
    build_episodes_symbol,
)
from research.liquidity_oracle_atlas.experiment_path0_episode_path_memory_v1 import (  # noqa: E402
    BIT_NAMES, BIT_MASKS, ece_binary, make_pipeline, binary_logloss,
    episode_identity_hash, REVIEWER_FROZEN_EPISODE_HASH,
)
from research.liquidity_oracle_atlas.experiment_state1_provenance_decomposition_v1 import (  # noqa: E402
    group_provenance,
)
from research.liquidity_oracle_atlas.experiment_state11_functional_provenance_v1 import (  # noqa: E402
    FUNC_PROV,
)
from panji_indicators import compute_smc_pine  # canonical SMC (frozen semantics)

REPL_BLOCK = "TB3"
EPS = 1e-9
BOOTSTRAP_REPS = 1000

# --------------------------------------------------------------- feature names
GEOM_START = ["start_up_distance_R", "start_down_distance_R",
              "start_width_R", "start_log_ratio"]
GEOM_CUR = ["cur_up_distance_R", "cur_down_distance_R",
            "cur_width_R", "cur_log_ratio"]
PATH_COLS = ["path_total_variation_R", "path_max_up_excursion_R",
             "path_max_down_excursion_R", "path_direction_change_rate",
             "path_last_return_R", "path_current_bar_range_R"]
SMC_EVENT_BITS = ["bos_internal_bull", "bos_internal_bear",
                  "choch_internal_bull", "choch_internal_bear",
                  "bos_swing_bull", "bos_swing_bear",
                  "choch_swing_bull", "choch_swing_bear"]
SMC_BIAS = ["internal_bias", "swing_bias"]
OB_COLS = ["ob_created_bull", "ob_created_bear", "ob_entered_bull",
           "ob_entered_bear", "ob_mitigated_bull", "ob_mitigated_bear",
           "active_internal_ob_count"]
SMC_ALL = SMC_EVENT_BITS + SMC_BIAS + OB_COLS
CAT = ["prev_event_mask"]
ELAPSED = ["elapsed_log"]

# full numeric feature superset == M4 (every model is a strict prefix)
NUM_ALL = (GEOM_START + FUNC_PROV + ELAPSED + GEOM_CUR + PATH_COLS
           + SMC_EVENT_BITS + SMC_BIAS + OB_COLS)
assert len(NUM_ALL) == 42, len(NUM_ALL)

M0_NUM = GEOM_START + FUNC_PROV + ELAPSED
M1_NUM = M0_NUM + GEOM_CUR
M2_NUM = M1_NUM + PATH_COLS
M3_NUM = M2_NUM + SMC_EVENT_BITS + SMC_BIAS
M4_NUM = M3_NUM + OB_COLS
assert M4_NUM == NUM_ALL
assert set(M0_NUM) < set(M1_NUM) < set(M2_NUM) < set(M3_NUM) < set(M4_NUM)

MODELS = {
    "M0_START_STATE": (M0_NUM, CAT),
    "M1_CURRENT_5M_SNAPSHOT": (M1_NUM, CAT),
    "M2_INTRA_EPISODE_PATH": (M2_NUM, CAT),
    "M3_SMC_STRUCTURE": (M3_NUM, CAT),
    "M4_OB_LIFECYCLE": (M4_NUM, CAT),
}
M0, M1, M2, M3, M4 = MODELS.keys()

COMPARISONS = [
    ("M1-M0", "BAR_STATE_UPDATE", M1, M0),
    ("M2-M1", "INTRA_EPISODE_PATH_MEMORY", M2, M1),
    ("M3-M2", "SMC_STRUCTURE_INCREMENT", M3, M2),
    ("M4-M3", "OB_LIFECYCLE_INCREMENT", M4, M3),
]

WINDOWS = [
    dict(name="A_TB1_to_TB2", train=[TRAIN_BLOCK], eval=TEST_BLOCK,
         seed=20260920),
    dict(name="B_TB1TB2_to_TB3", train=[TRAIN_BLOCK, TEST_BLOCK],
         eval=REPL_BLOCK, seed=20260921),
]


# ===========================================================================
# canonical functional provenance (10 vars) — 与 STATE-1.1 同公式，逐 bar 重算
# ===========================================================================
def func_prov_vec(pu, pl):
    uo, un, ua = pu
    lo, ln, la = pl
    return np.array([
        np.log1p(uo), np.log1p(un), float(un == 0), float(uo == 0), float(ua),
        np.log1p(lo), np.log1p(ln), float(ln == 0), float(lo == 0), float(la),
    ], dtype=float)


# ===========================================================================
# causal running path (section 5) — 仅用 s..t 已知信息，t 为 observation bar
# ===========================================================================
def build_running_path(h, l, c, s, e, atr0):
    a0 = max(float(atr0), EPS)
    n = e - s
    tv = np.zeros(n)
    mh = np.zeros(n)
    ml = np.zeros(n)
    dcr = np.zeros(n)
    last_ret = np.zeros(n)
    cbr = np.zeros(n)
    prev_sign = 0
    dir_ch = 0
    n_nz = 0
    mh_cur = float(h[s])
    ml_cur = float(l[s])
    for k in range(n):
        t = s + k
        if k > 0:
            delta = float(c[t] - c[t - 1])
            tv[k] = tv[k - 1] + abs(delta)
            sign = int(np.sign(delta))
            if sign != 0:
                if prev_sign != 0 and sign != prev_sign:
                    dir_ch += 1
                prev_sign = sign
                n_nz += 1
        else:
            delta = 0.0
        mh_cur = max(mh_cur, float(h[t]))
        ml_cur = min(ml_cur, float(l[t]))
        mh[k] = mh_cur
        ml[k] = ml_cur
        dcr[k] = dir_ch / max(n_nz - 1, 1) if n_nz >= 2 else 0.0
        last_ret[k] = delta / a0
        cbr[k] = (float(h[t]) - float(l[t])) / a0
    tv /= a0
    mh = (mh - float(c[s])) / a0
    ml = (float(c[s]) - ml) / a0
    return dict(
        path_total_variation_R=tv,
        path_max_up_excursion_R=mh,
        path_max_down_excursion_R=ml,
        path_direction_change_rate=dcr,
        path_last_return_R=last_ret,
        path_current_bar_range_R=cbr,
    )


# ===========================================================================
# SMC（canonical，只跑一次 per symbol，截断到 TB3）
# ===========================================================================
def build_smc_arrays(opens, highs, lows, closes, times):
    n = len(closes)
    out = compute_smc_pine(opens, highs, lows, closes, times,
                           params=None, emit_timeline=True)
    arr = {k: np.zeros(n, dtype=np.int64)
           for k in SMC_EVENT_BITS + OB_COLS}
    arr["internal_bias"] = np.zeros(n, dtype=np.int64)
    arr["swing_bias"] = np.zeros(n, dtype=np.int64)
    arr["active_internal_ob_count"] = np.zeros(n, dtype=np.int64)

    for ev in out["events"]:
        i = int(ev["confirmed_index"])
        if not (0 <= i < n):
            raise RuntimeError(f"STOP_SMC_EVENT_INDEX_OOB: {i} n={n}")
        typ = str(ev["type"]).lower()
        level = "internal" if bool(ev["internal"]) else "swing"
        side = "bull" if bool(ev["bullish"]) else "bear"
        key = f"{typ}_{level}_{side}"
        if key in arr:
            arr[key][i] = 1

    for ev in out["ob_lifecycle_events"]:
        if not bool(ev["internal"]):
            continue  # 第一轮只使用 canonical 默认启用的 internal OB
        side = "bull" if int(ev["bias"]) > 0 else "bear"
        typ = str(ev["type"])
        if typ == "OB_CREATED":
            i = int(ev["confirmed_index"]); key = f"ob_created_{side}"
        elif typ == "OB_ENTERED":
            i = int(ev["enter_index"]); key = f"ob_entered_{side}"
        elif typ == "OB_MITIGATED":
            i = int(ev["mitigated_index"]); key = f"ob_mitigated_{side}"
        else:
            continue
        if not (0 <= i < n):
            raise RuntimeError(f"STOP_SMC_OB_INDEX_OOB: {typ} {i} n={n}")
        if key in arr:
            arr[key][i] = 1

    for r in out["state_timeline"]:
        i = int(r["bar_index"])
        if 0 <= i < n:
            arr["internal_bias"][i] = int(r["internal_bias"])
            arr["swing_bias"][i] = int(r["swing_bias"])
            arr["active_internal_ob_count"][i] = int(r["active_internal_ob_count"])
    return arr


def causality_check(sym, full_arr, opens, highs, lows, closes, times, tb3_end):
    """15 symbols x 5 cutpoints = 75 prefix checks (section 16)."""
    cutpoints = [max(2, int(frac * tb3_end))
                 for frac in (0.2, 0.4, 0.6, 0.8, 0.95)]
    rows = []
    for t in cutpoints:
        pref = build_smc_arrays(opens[:t + 1], highs[:t + 1], lows[:t + 1],
                                closes[:t + 1], times[:t + 1])
        ok = True
        bad = []
        for k in SMC_ALL:
            if int(full_arr[k][t]) != int(pref[k][t]):
                ok = False
                bad.append(k)
        rows.append(dict(symbol=sym, cutpoint=t, pass_=bool(ok),
                         mismatched=";".join(bad)))
    return rows


# ===========================================================================
# per-episode NLL（section 11）
# ===========================================================================
def episode_nll(p_h, p_mask, h, mask, ep_id):
    ph = np.clip(p_h, 1e-15, 1 - 1e-15)
    row_loss = -(h * np.log(ph) + (1 - h) * np.log1p(-ph))
    term = np.flatnonzero(h == 1)
    yi = mask_index(mask[term])
    row_loss[term] += -np.log(np.clip(
        p_mask[np.arange(len(term)), yi], 1e-15, 1.0))
    uniq, inv = np.unique(ep_id, return_inverse=True)
    per = np.bincount(inv, weights=row_loss, minlength=len(uniq))
    return uniq, per


def boot_delta(delta, day, seed, reps=BOOTSTRAP_REPS):
    uniq = np.unique(day)
    pos = np.searchsorted(uniq, day)
    nd = len(uniq)
    cnt = np.bincount(pos, minlength=nd)
    keep = cnt > 0
    nk = int(keep.sum())
    dv = np.bincount(pos, weights=delta, minlength=nd)[keep] / cnt[keep]
    rng = np.random.default_rng(seed)
    b = np.empty(reps)
    for i in range(reps):
        b[i] = dv[rng.integers(0, nk, nk)].mean()
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# ===========================================================================
# fit + evaluate one model on one window
# ===========================================================================
def fit_eval(model_name, num_cols, cat_cols, tr, ev):
    cols = num_cols + cat_cols
    pipe = make_pipeline(num_cols, cat_cols)
    pipe.fit(tr[cols], tr["hazard"].to_numpy())
    pre = pipe.named_steps["pre"]
    p_h = pipe.predict_proba(ev[cols])[:, 1]

    Xtr = densify(pre.transform(tr[cols]))
    Xev = densify(pre.transform(ev[cols]))
    tt = tr["hazard"].to_numpy() == 1
    mtr = tr["target_mask"].to_numpy()[tt].astype(np.int64)
    theta, aud = fit_conditional_crf(Xtr[tt], mtr, with_pairs=True)

    te = ev["hazard"].to_numpy() == 1
    p_mask = predict_mask_prob(theta, Xev[te], with_pairs=True)
    return dict(p_h=p_h, p_mask=p_mask, te=te, theta=theta, aud=aud)


def model_metrics(ev, p_h, p_mask, te):
    h = ev["hazard"].to_numpy().astype(np.int64)
    ph = np.clip(p_h, 1e-15, 1 - 1e-15)
    haz_nll = float(-np.mean(h * np.log(ph) + (1 - h) * np.log1p(-ph)))
    haz_brier = float(np.mean((p_h - h) ** 2))

    m = ev["target_mask"].to_numpy()[te].astype(np.int64)
    yi = mask_index(m)
    ep_nll = float(-np.mean(np.log(np.clip(
        p_mask[np.arange(len(yi)), yi], 1e-15, 1.0))))

    y_bits = np.stack([((m & bit) != 0).astype(np.int64)
                       for bit in BIT_MASKS], axis=1)
    marg = p_mask @ np.array(
        [[(mv >> k) & 1 for k in range(4)] for mv in range(1, 16)],
        dtype=float)
    ece = float(np.mean([ece_binary(y_bits[:, k], marg[:, k])
                         for k in range(4)]))

    uniq, per = episode_nll(p_h, p_mask, h,
                            ev["target_mask"].to_numpy().astype(np.int64),
                            ev["episode_id"].to_numpy())
    mean_ep = float(per.mean())

    # duration calibration (non-gating): expected vs actual mean episode length.
    # EXPECTED duration per episode = sum_k prod_{j<k}(1 - p_h_j) (reset each episode).
    ph_arr = np.clip(p_h, 0, 1 - 1e-15)
    df2 = pd.DataFrame({"ep": ev["episode_id"].to_numpy(),
                        "bt": ev["bar_t"].to_numpy(), "p": ph_arr})
    df2 = df2.sort_values(["ep", "bt"])
    exp_dur = []
    for _, g in df2.groupby("ep"):
        ps = g["p"].to_numpy()
        surv = 1.0
        tot = 0.0
        for p in ps:
            tot += surv
            surv *= (1.0 - p)
        exp_dur.append(tot)
    mean_pred = float(np.mean(exp_dur))
    actual = (ev.groupby("episode_id")["bar_t"].max()
              - ev.groupby("episode_id")["start_bar"].first()).to_numpy() + 1
    mean_actual = float(np.mean(actual))
    return dict(hazard_nll=haz_nll, hazard_brier=haz_brier,
                endpoint_joint_nll=ep_nll, marginal_bit_ece=ece,
                mean_episode_nll=mean_ep, n_episodes=int(len(uniq)),
                n_obs=int(len(ev)), mean_pred_duration=mean_pred,
                mean_actual_duration=mean_actual)


def _aggregate(values, ep_id):
    uniq, inv = np.unique(ep_id, return_inverse=True)
    return np.bincount(inv, weights=values, minlength=len(uniq))


# ===========================================================================
# build bar-level observations for one symbol
# episodes: list of episode dicts for this symbol (frozen TB1/TB2 parquet rows
#           + freshly built TB3 episodes), sorted later by start_bar.
# ===========================================================================
def build_symbol_obs(sym, bars, grp, smc_arr, episodes, tb3_end, start_id=0):
    c = bars["c"]; h = bars["h"]; l = bars["l"]; atr = bars["atr"]
    td = np.asarray(bars["td"]).astype("datetime64[D]")

    # previous episode endpoint mask (same symbol, sorted by start)
    eps_sorted = sorted(episodes, key=lambda e: int(e["start_bar"]))
    prev_mask = {}
    for i, e in enumerate(eps_sorted):
        s = int(e["start_bar"])
        prev_mask[s] = int(eps_sorted[i - 1]["event_mask"]) if i > 0 else 0

    parts = {col: [] for col in NUM_ALL + CAT}
    meta_cols = ["symbol", "episode_id", "start_bar", "bar_t", "block",
                 "episode_start_day", "hazard", "target_mask"]
    meta = {col: [] for col in meta_cols}

    ep_id_counter = start_id
    n_zero_dur = 0
    n_censor = 0
    n_atr0 = 0
    for e in eps_sorted:
        s = int(e["start_bar"]); ed = int(e["end_bar"])
        blk = str(e["start_block"])
        if str(e["end_block"]) != blk:
            continue  # 跨 block 的 episode 排除，保证 train/eval 不泄漏
        mask = int(e["event_mask"])
        if mask == 0:
            n_censor += 1
            continue  # censor（无 structural endpoint）排除；endpoint head 未定义
        if ed <= s:
            n_zero_dur += 1
            continue
        U = float(e["start_upper_price"]); D = float(e["start_lower_price"])
        atr0 = float(atr[s])
        if atr0 <= 0:
            n_atr0 += 1
            continue

        pu = group_provenance(grp, int(e["start_upper_group"]), s)
        pl = group_provenance(grp, int(e["start_lower_group"]), s)
        if pu is None or pl is None:
            continue
        fp = func_prov_vec(pu, pl)

        n = ed - s
        tt = np.arange(s, ed)  # observation bars t = s .. ed-1
        cu = np.atleast_1d((U - c[tt]) / atr0)
        cd = np.atleast_1d((c[tt] - D) / atr0)
        cw = np.full(n, (U - D) / atr0)  # width 不依赖 t，须广播到 n
        clr = np.atleast_1d(np.log((U - c[tt] + EPS) / (c[tt] - D + EPS)))
        path = build_running_path(h, l, c, s, ed, atr0)
        path = {k: np.atleast_1d(np.asarray(v, dtype=float)) for k, v in path.items()}
        haz = np.zeros(n, dtype=np.int64); haz[-1] = 1
        tmask = np.zeros(n, dtype=np.int64); tmask[-1] = mask
        el = np.log1p(np.arange(n))

        start_geom = np.array([
            (U - c[s]) / atr0, (c[s] - D) / atr0, (U - D) / atr0,
            np.log((U - c[s] + EPS) / (c[s] - D + EPS))], dtype=float)
        pvm = prev_mask[s]

        # -- accumulate per-column arrays --
        for j, name in enumerate(GEOM_START):
            parts[name].append(np.full(n, start_geom[j]))
        for j, name in enumerate(FUNC_PROV):
            parts[name].append(np.full(n, fp[j]))
        parts["elapsed_log"].append(el)
        for j, name in enumerate(GEOM_CUR):
            src = [cu, cd, cw, clr][j]
            parts[name].append(src.astype(float))
        for name in PATH_COLS:
            parts[name].append(path[name].astype(float))
        for name in SMC_EVENT_BITS + SMC_BIAS + OB_COLS:
            parts[name].append(smc_arr[name][s:ed].astype(float))
        parts["prev_event_mask"].append(np.full(n, pvm, dtype=np.int64))

        meta["symbol"].append([sym] * n)
        meta["episode_id"].append([ep_id_counter] * n)
        meta["start_bar"].append([s] * n)
        meta["bar_t"].append(tt.tolist())
        meta["block"].append([blk] * n)
        meta["episode_start_day"].append([str(td[s])] * n)
        meta["hazard"].append(haz.tolist())
        meta["target_mask"].append(tmask.tolist())
        ep_id_counter += 1

    if ep_id_counter == 0:
        return None, dict(n_zero_dur=n_zero_dur, n_censor=n_censor, n_atr0=n_atr0)
    df_num = pd.DataFrame({col: np.concatenate(parts[col])
                           for col in NUM_ALL + CAT})
    df_meta = pd.DataFrame({col: np.concatenate(
        [np.array(v, dtype=object if col in ("symbol", "block",
         "episode_start_day") else int) for v in meta[col]])
        for col in meta_cols})
    df = pd.concat([df_meta, df_num], axis=1)
    audit = dict(n_episodes=ep_id_counter, n_bar_rows=int(len(df)),
                 n_zero_dur=n_zero_dur, n_censor=n_censor, n_atr0=n_atr0)
    return df, audit, ep_id_counter


# ===========================================================================
# main
# ===========================================================================
def main():
    t_total = time.perf_counter()
    timing = {}

    # ---------------------------------------------------- load + block grid
    t0 = time.perf_counter()
    bars_by_sym = {s: load_raw_bars(s) for s in FULL_UNIV}
    all_days, day_block_code, boundaries = build_blocks(bars_by_sym)
    block_names = np.array([f"TB{i + 1}" for i in range(4)], dtype=object)
    timing["cache_load_seconds"] = round(time.perf_counter() - t0, 2)

    sym_info = {}
    for s in FULL_UNIV:
        b = bars_by_sym[s]
        td = np.asarray(b["td"]).astype("datetime64[D]")
        code = day_block_code[np.searchsorted(all_days, td)]
        tb3_end = int(np.flatnonzero(code <= 2)[-1])
        sym_info[s] = dict(bars=b, code=code, tb3_end=tb3_end)
    del bars_by_sym

    # ------------------------------------------------ frozen TB1/TB2 episodes
    t_h = time.perf_counter()
    ep0_df = pd.read_parquet(CACHE / "episode0_episodes.parquet")
    ep_hash = episode_identity_hash(ep0_df)
    if ep_hash != REVIEWER_FROZEN_EPISODE_HASH:
        raise SystemExit(
            f"STOP_PGM_BAR0_EPISODE_HASH_MISMATCH: {ep_hash} != "
            f"{REVIEWER_FROZEN_EPISODE_HASH}")
    timing["hash_seconds"] = round(time.perf_counter() - t_h, 2)

    # ---------------------------------------------------- per-symbol build
    t_build = time.perf_counter()
    obs_parts = []
    causality_rows = []
    smc_counts = []
    sym_audit = {}
    ep_id = 0  # globally unique episode counter across all symbols
    for s in FULL_UNIV:
        info = sym_info[s]
        b = info["bars"]; code = info["code"]; tb3_end = info["tb3_end"]
        N = tb3_end + 1
        times = [pd.Timestamp(x).isoformat() for x in b["t"][:N]]
        smc = build_smc_arrays(b["o"][:N].tolist(), b["h"][:N].tolist(),
                               b["l"][:N].tolist(), b["c"][:N].tolist(), times)
        # smc 由 length-N 截断数组构建，所有 index 天然 < N（TB4 已截断）

        causality_rows += causality_check(
            s, smc, b["o"][:N].tolist(), b["h"][:N].tolist(),
            b["l"][:N].tolist(), b["c"][:N].tolist(), times, tb3_end)

        cnt = dict(symbol=s)
        for k in SMC_ALL:
            cnt[k] = int(smc[k].sum())
        smc_counts.append(cnt)

        seq, grp = load_seq(s)
        # TB1/TB2: frozen parquet (identity guaranteed); TB3: build fresh
        ep0_sym = ep0_df[ep0_df["symbol"] == s].to_dict("records")
        block_of_bar = np.array(["TB1", "TB2", "TB3", "TB4"], dtype=object)[code]
        bar_end_time = (np.asarray(b["t"]).astype("datetime64[ns]")
                        + np.timedelta64(5, "m"))
        eps_tb3, _, _ = build_episodes_symbol(
            s, seq, grp, b["h"], b["l"], b["disc"], b["c"], block_of_bar,
            bar_end_time, tb3_end)
        eps_tb3 = [e for e in eps_tb3 if str(e["start_block"]) == REPL_BLOCK
                   and str(e["end_block"]) == REPL_BLOCK]
        combined = ep0_sym + eps_tb3  # prev_mask links across both blocks

        df, audit, ep_id = build_symbol_obs(s, b, grp, smc, combined, tb3_end, ep_id)
        sym_audit[s] = audit
        if df is not None:
            obs_parts.append(df)

    obs = pd.concat(obs_parts, ignore_index=True)
    timing["symbol_build_seconds"] = round(time.perf_counter() - t_build, 2)

    # causality verdict
    causality_df = pd.DataFrame(causality_rows)
    n_pass = int(causality_df["pass_"].sum())
    if n_pass != len(causality_df):
        bad = causality_df[~causality_df["pass_"]]
        raise SystemExit(
            f"STOP_SMC_PREFIX_CAUSALITY_FAIL: {len(bad)} mismatches\n"
            f"{bad.to_string()}")

    # window flag: TB4 never contributes
    obs.to_parquet(CACHE / "pgm_bar0_samples.parquet", index=False)

    # ------------------------------------------------------------ windows
    res = []
    all_per_ep = {}
    for w in WINDOWS:
        tr = obs[obs["block"].isin(w["train"])].reset_index(drop=True)
        ev = obs[obs["block"] == w["eval"]].reset_index(drop=True)
        if not len(tr) or not len(ev):
            raise SystemExit(f"STOP_PGM_BAR0_EMPTY_SPLIT: {w['name']}")
        if w["eval"] in set(tr["block"]):
            raise SystemExit(f"STOP_PGM_BAR0_EVAL_IN_FIT: {w['name']}")

        per_ep = {}
        model_out = {}
        for name, (num_cols, cat_cols) in MODELS.items():
            out = fit_eval(name, num_cols, cat_cols, tr, ev)
            met = model_metrics(ev, out["p_h"], out["p_mask"], out["te"])
            model_out[name] = dict(metrics=met, aud=out["aud"])
            h = ev["hazard"].to_numpy().astype(np.int64)
            uniq, per = episode_nll(
                out["p_h"], out["p_mask"], h,
                ev["target_mask"].to_numpy().astype(np.int64),
                ev["episode_id"].to_numpy())
            per_ep[name] = (uniq, per)

        # bootstrap comparisons
        boots = []
        by_sym = []
        for label, gate, hi, lo in COMPARISONS:
            uniq_h, per_h = per_ep[hi]
            uniq_l, per_l = per_ep[lo]
            if not np.array_equal(uniq_h, uniq_l):
                raise SystemExit(
                    f"STOP_PGM_BAR0_EPISODE_MISALIGN: {label}")
            delta = per_h - per_l
            day = ev.groupby("episode_id")["episode_start_day"].first()
            day = day.reindex(pd.Index(uniq_h)).to_numpy()
            a, b = boot_delta(delta, day, w["seed"]
                             + ["M1-M0", "M2-M1", "M3-M2", "M4-M3"].index(label))
            verdict = ("CI_below_zero" if b < 0
                       else "CI_above_zero" if a > 0 else "CI_contains_zero")
            # align by-symbol
            sym = ev.groupby("episode_id")["symbol"].first()
            sym = sym.reindex(pd.Index(uniq_h)).to_numpy()
            for s_ in FULL_UNIV:
                idx = sym == s_
                if idx.any():
                    d = delta[idx]
                    by_sym.append(dict(window=w["name"], comparison=label,
                                       symbol=s_, n=int(idx.sum()),
                                       mean_delta=float(d.mean()),
                                       n_negative=int((d < 0).sum()),
                                       n_positive=int((d > 0).sum())))
            boots.append(dict(window=w["name"], comparison=label, gate=gate,
                              model_hi=hi, model_lo=lo,
                              delta_sample_mean=float(delta.mean()),
                              ci_lo=a, ci_hi=b, verdict=verdict))
        res.append(dict(win=w, model_out=model_out, boots=boots,
                        by_sym=by_sym, n_train=int(len(tr)),
                        n_eval=int(len(ev))))

    # ------------------------------------------------------------ outputs
    mtab_rows = []
    for r in res:
        w = r["win"]["name"]
        for name, mo in r["model_out"].items():
            m = mo["metrics"]
            mtab_rows.append(dict(window=w, model=name, **m))
    mtab = pd.DataFrame(mtab_rows)
    mtab.to_csv(OUT / "pgm_bar0_model_metrics.csv", index=False)

    boots_df = pd.DataFrame([b for r in res for b in r["boots"]])
    boots_df.to_csv(OUT / "pgm_bar0_bootstrap.csv", index=False)

    bysym_df = pd.DataFrame([b for r in res for b in r["by_sym"]])
    bysym_df.to_csv(OUT / "pgm_bar0_by_symbol.csv", index=False)

    pd.DataFrame(smc_counts).to_csv(OUT / "pgm_bar0_smc_event_counts.csv",
                                    index=False)
    causality_df.to_csv(OUT / "pgm_bar0_causality_audit.csv", index=False)

    opt_rows = []
    for r in res:
        w = r["win"]["name"]
        for name, mo in r["model_out"].items():
            a = mo["aud"]
            opt_rows.append(dict(window=w, model=name, n_params=int(a["n_params"]),
                                n_iter=int(a["n_iter"]),
                                final_fun=float(a["final_fun"]),
                                success=bool(a["success"]),
                                elapsed_seconds=round(a["elapsed_seconds"], 2)))
    pd.DataFrame(opt_rows).to_csv(OUT / "pgm_bar0_optimizer_audit.csv",
                                 index=False)

    # ------------------------------------------------------------ verdict
    verdict = {}
    for label, gate, hi, lo in COMPARISONS:
        ws = boots_df[boots_df["comparison"] == label]
        both = bool((ws["ci_hi"] < 0).all())
        verdict[gate] = dict(
            supported=both,
            windows={r["win"]["name"]: dict(
                ci_hi=float(ws[ws["window"] == r["win"]["name"]]["ci_hi"].iloc[0]))
                for r in res})

    # ------------------------------------------------------------ sample audit
    n_ep_total = int(obs["episode_id"].nunique())
    sample_audit = dict(
        n_episodes=int(n_ep_total),
        n_bar_rows=int(len(obs)),
        n_terminals=int(obs["hazard"].sum()),
        episode_identity_sha256=ep_hash,
        episode_hash_matches_frozen=bool(
            ep_hash == REVIEWER_FROZEN_EPISODE_HASH),
        frozen_episode_hash=REVIEWER_FROZEN_EPISODE_HASH,
        tb4_analytically_used=False,
        per_symbol={s: {k: v for k, v in a.items()}
                    for s, a in sym_audit.items()},
        smc_causality=dict(n_checks=int(len(causality_df)),
                           n_pass=n_pass,
                           all_pass=bool(n_pass == len(causality_df))),
    )
    (OUT / "pgm_bar0_sample_audit.json").write_text(
        json.dumps(sample_audit, indent=2, default=str))

    summary = dict(
        experiment="PGM-BAR-0 5m dynamic path + SMC transition increment",
        base="2ccb912ea00ba0fcf3f0bc1d694cb89eae169668",
        question=("does 5m dynamic update / intra-episode path / SMC structure "
                  "/ OB lifecycle each add stable episode-NLL increment over the "
                  "frozen 15-mask pairwise CRF endpoint head?"),
        models={k: dict(n_numeric=len(v[0]), n_categorical=len(v[1]),
                        numeric=v[0], categorical=v[1])
                for k, v in MODELS.items()},
        nested="M0⊂M1⊂M2⊂M3⊂M4 (strict)",
        windows=[dict(name=w["name"], train=w["train"], eval=w["eval"],
                      seed=w["seed"]) for w in WINDOWS],
        primary_metric="mean episode NLL",
        endpoint_head="15-mask pairwise conditional CRF (PGM-0, frozen)",
        smc_source="panji_indicators.compute_smc_pine canonical defaults; "
                   "internal OB only (swing OB disabled by canonical default)",
        tb4_analytically_used=False,
        sample_audit=sample_audit,
        bootstrap_reps=BOOTSTRAP_REPS,
        model_metrics={r["win"]["name"]: {name: mo["metrics"]
                       for name, mo in r["model_out"].items()}
                       for r in res},
        bootstrap=boots_df.to_dict(orient="records"),
        by_symbol=dict(
            n_rows=int(len(bysym_df)),
            windows={w["name"]: {label: dict(
                n_negative=int(((bysym_df["window"] == w["name"])
                                & (bysym_df["comparison"] == label)
                                & (bysym_df["n_negative"] > 0)).sum()),
                n_positive=int(((bysym_df["window"] == w["name"])
                                & (bysym_df["comparison"] == label)
                                & (bysym_df["n_positive"] > 0)).sum()))
                for label, gate, hi, lo in COMPARISONS}
                for w in WINDOWS}),
        verdict=verdict,
        gates={
            "BAR_STATE_UPDATE": "M1-M0 CI_hi<0 in both windows",
            "INTRA_EPISODE_PATH_MEMORY": "M2-M1 CI_hi<0 in both windows",
            "SMC_STRUCTURE_INCREMENT": "M3-M2 CI_hi<0 in both windows",
            "OB_LIFECYCLE_INCREMENT": "M4-M3 CI_hi<0 in both windows",
        },
        interpretation_limits=(
            "PGM-BAR-0 only tests whether dynamic 5m / path / SMC / OB features "
            "reduce episode NLL beyond the frozen endpoint CRF head. It does NOT "
            "authorise RL, PnL, latent state, HMM/HSMM, new Sweep definitions, "
            "SMC parameter tuning, or tradability."),
    )
    (OUT / "pgm_bar0_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    timing["total_seconds"] = round(time.perf_counter() - t_total, 2)
    summary["timing"] = timing
    (OUT / "pgm_bar0_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    print(f"[SAMPLE] n_episodes={n_ep_total} n_bar_rows={len(obs)} "
          f"hash_match={sample_audit['episode_hash_matches_frozen']}")
    print(f"[CAUSALITY] {n_pass}/{len(causality_df)} PASS")
    print(f"[MODEL METRICS]\n{mtab.to_string(index=False)}")
    print(f"[BOOTSTRAP]\n{boots_df.to_string(index=False)}")
    print(f"[VERDICT] {verdict}")
    print(f"[TIMING] {timing}")
    print(f"[DONE] -> {OUT}")


if __name__ == "__main__":
    main()
