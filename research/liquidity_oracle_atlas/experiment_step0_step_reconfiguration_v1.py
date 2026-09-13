"""STEP-0 — Local Step Reconfiguration Audit

===========================================================================
问题（本轮唯一问题）
===========================================================================
在 decision time 冻结 nearest upper/lower liquidity 之后，在 frozen pair
被 penetration **之前**，新的 causal liquidity activation 是否经常使当前的
nearest upper/lower pair 发生重构？

    t0 : 冻结 U0 / D0
    j in (t0, resolution_bar - 1] : 用 LOCAL-0 完全相同的 causal active /
         nearest-pair contract 重算 U_j / D_j
    U_j != U0  -> UPPER_RECONFIG
    D_j != D0  -> LOWER_RECONFIG
    同 bar 双侧 -> BOTH

本轮是 **pure structural audit**：不训练模型，不设 PASS/FAIL 阈值。
它只决定以后 PATH episode 应该如何切段。

STEP-0 does NOT establish predictive value. 这些 future reconfiguration
字段禁止进入任何当前-step prediction model。

===========================================================================
冻结边界
===========================================================================
* 不改 activation / penetration / expiry / discontinuity / nearest-pair
  定义；直接复用 LOCAL-0 的冻结 kernel。
* corrected lifecycle 用冻结函数 `build_corrected_lifecycle` 重新推导
  （纯函数、确定性、输入仅为 master + bars），并与 frozen sample cache
  做 pair-key parity（100% 必须一致，否则 STOP）。
* 只用 TB1 / TB2。TB3 / TB4 仅验证其存在，不计算任何 STEP-0 统计。
* 禁止 PATH feature / Path Signature / SMC / liquidity type / HMM / PGM /
  RL / PnL。

===========================================================================
效率
===========================================================================
不做 state x future 无界扫描。核心是区间 change-point：
先对每个 symbol 的 full-bar pair sequence 求 change positions，
再用 searchsorted 一次性回答所有 state 的 "first change after t0" 与
"transitions in (t0, T-1]"。比较量 = sum(duration)，千万级。

===========================================================================
输出
===========================================================================
    step0_summary.json
    step0_duration_bins.csv
    step0_by_symbol.csv
    step0_by_outcome.csv
    step0_reconfig_types.csv
    step0_non_inward_examples.csv
    step0_first_reconfig_examples.csv
（per-state audit 存 gitignored cache，不提交）
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

# --- 复用 LOCAL-0 的冻结 kernel（不修改其中任何定义） -------------------
from research.liquidity_oracle_atlas.run_5m_graph_probability_v1 import (  # noqa: E402
    load_raw_bars,
    load_master,
)
from research.liquidity_oracle_atlas.experiment_local_liquidity_transition_v0 import (  # noqa: E402
    FULL_UNIV, UP, DOWN, AMBIGUOUS, RESOLVED_CODES, RIGHT_CENSOR_CODES,
    TRAIN_BLOCK, TEST_BLOCK, OUT, CACHE, I64MAX, I64MIN,
    build_corrected_lifecycle, build_level_groups,
    active_level_groups_chunk, nearest_active_pair_chunk_v2,
)

# 534de62e 冻结基线
EXPECT = dict(tb1_resolved=114898, tb2_resolved=124612)

# reconfig side codes
S_NONE, S_UPPER, S_LOWER, S_BOTH = 0, 1, 2, 3
SIDE_NAMES = np.array(["NONE", "UPPER_ONLY", "LOWER_ONLY", "BOTH"], dtype=object)

DUR_BINS = [(1, 5), (6, 12), (13, 24), (25, 48), (49, 96), (97, 10 ** 9)]
DUR_LABELS = ["1-5", "6-12", "13-24", "25-48", "49-96", "97+"]


# ===========================================================================
# full-bar pair sequence (per symbol)
# ===========================================================================
def group_max_activation(bar_idx: np.ndarray, group_idx: np.ndarray,
                         info: dict) -> np.ndarray:
    """selected group 内 **active** identities 的最大 activation_bar。

    用于 STEP-0 §13 的 activation provenance：新 group 是否确实在
    decision 之后才 causal available。
    """
    i = np.asarray(bar_idx, dtype=np.int64)[:, None]
    ab = info["activation_bar"]
    eb = info["expiry_bar"]
    pb = info["penetration_bar"]
    gid = info["id_group"]
    g = np.maximum(np.asarray(group_idx, dtype=np.int64), 0)
    ok = (np.asarray(group_idx, dtype=np.int64) >= 0)[:, None]

    active_id = ((ab[None, :] <= i) & (i < eb[None, :])
                 & ((pb[None, :] < 0) | (i < pb[None, :])))
    m = active_id & (gid[None, :] == g[:, None]) & ok
    vals = np.where(m, ab[None, :], I64MIN)
    out = np.max(vals, axis=1)
    return np.where(np.asarray(group_idx) >= 0, out, -1).astype(np.int64)


def build_bar_pair_arrays(sym: str, master_sym: pd.DataFrame, bars: dict,
                          lc: dict) -> dict:
    """对每个 5m bar 计算动态 nearest pair（与 LOCAL-0 完全同 contract）。"""
    info = build_level_groups(master_sym, lc, bars)
    n = int(bars["n"])
    close = np.asarray(bars["c"], dtype=np.float64)
    bar_idx = np.arange(n, dtype=np.int64)

    up_g = np.full(n, -1, dtype=np.int64)
    dn_g = np.full(n, -1, dtype=np.int64)
    up_px = np.full(n, np.nan, dtype=np.float64)
    dn_px = np.full(n, np.nan, dtype=np.float64)
    up_act = np.full(n, -1, dtype=np.int64)
    dn_act = np.full(n, -1, dtype=np.int64)

    chunk = 256
    for lo in range(0, n, chunk):
        sl = slice(lo, min(lo + chunk, n))
        bi = bar_idx[sl]
        pair = nearest_active_pair_chunk_v2(bi, close[sl], info)
        up_g[sl] = pair["upper_group"]
        dn_g[sl] = pair["lower_group"]
        up_px[sl] = pair["upper_price"]
        dn_px[sl] = pair["lower_price"]
        up_act[sl] = group_max_activation(bi, pair["upper_group"], info)
        dn_act[sl] = group_max_activation(bi, pair["lower_group"], info)
    return dict(upper_group=up_g, lower_group=dn_g, upper_price=up_px,
                lower_price=dn_px, upper_max_act=up_act,
                lower_max_act=dn_act, n=n)


# ===========================================================================
# change-point primitives（scalar / vector 同一实现）
# ===========================================================================
def change_positions(arr: np.ndarray) -> np.ndarray:
    """j where arr[j] != arr[j-1]（j >= 1）。"""
    arr = np.asarray(arr)
    return np.flatnonzero(arr[1:] != arr[:-1]) + 1


def next_after(pos_sorted: np.ndarray, t: np.ndarray) -> np.ndarray:
    """first position in pos_sorted strictly > t；无则 -1。"""
    t = np.asarray(t, dtype=np.int64)
    k = np.searchsorted(pos_sorted, t, side="right")
    out = np.full(t.shape, -1, dtype=np.int64)
    hit = k < len(pos_sorted)
    out[hit] = pos_sorted[k[hit]]
    return out


def count_in_range(pos_sorted: np.ndarray, lo_excl: np.ndarray,
                   hi_incl: np.ndarray) -> np.ndarray:
    """count of positions p with lo_excl < p <= hi_incl。"""
    lo_excl = np.asarray(lo_excl, dtype=np.int64)
    hi_incl = np.asarray(hi_incl, dtype=np.int64)
    a = np.searchsorted(pos_sorted, lo_excl, side="right")
    b = np.searchsorted(pos_sorted, hi_incl, side="right")
    return (b - a).astype(np.int64)


# ===========================================================================
# STEP-0 state audit（向量化；合成测试用同一函数）
# ===========================================================================
def audit_states(seq: dict, close: np.ndarray,
                 decision_bar: np.ndarray,
                 resolution_bar: np.ndarray) -> dict:
    """对每个 frozen state 计算 reconfiguration 指标。

    seq: build_bar_pair_arrays 的输出
    close: (n,) 每个 bar 的 close
    decision_bar / resolution_bar: (S,)
    """
    U_g = seq["upper_group"]
    D_g = seq["lower_group"]
    U_px = seq["upper_price"]
    D_px = seq["lower_price"]
    U_act = seq["upper_max_act"]
    D_act = seq["lower_max_act"]

    t0 = np.asarray(decision_bar, dtype=np.int64)
    T = np.asarray(resolution_bar, dtype=np.int64)
    S = len(t0)

    up_changes = change_positions(U_g)
    dn_changes = change_positions(D_g)
    any_changes = np.union1d(up_changes, dn_changes)

    dur = T - t0
    hi = T - 1                       # interval (t0, T-1]

    f_up = next_after(up_changes, t0)
    f_dn = next_after(dn_changes, t0)
    f_up_ok = (f_up >= 0) & (f_up <= hi)
    f_dn_ok = (f_dn >= 0) & (f_dn <= hi)

    first = np.full(S, -1, dtype=np.int64)
    first[f_up_ok] = f_up[f_up_ok]
    both_mask = f_up_ok & f_dn_ok
    first[both_mask] = np.minimum(f_up[both_mask], f_dn[both_mask])
    first[~f_up_ok & f_dn_ok] = f_dn[~f_up_ok & f_dn_ok]

    reconfigured = first >= 0
    # side at the FIRST change bar
    side = np.full(S, S_NONE, dtype=np.int64)
    up_at = reconfigured & f_up_ok & (f_up == first)
    dn_at = reconfigured & f_dn_ok & (f_dn == first)
    side[up_at & ~dn_at] = S_UPPER
    side[dn_at & ~up_at] = S_LOWER
    side[up_at & dn_at] = S_BOTH

    # inward at the first change bar
    # 注意：upper/lower 的定义已保证 close_j < U_j 且 D_j < close_j，
    # 因此 inward 等价于 "新边界更近"。
    safe = np.where(reconfigured, first, 0)
    U0_px = U_px[t0]
    D0_px = D_px[t0]
    U1_px = U_px[safe]
    D1_px = D_px[safe]
    upper_inward = up_at & (U1_px < U0_px)
    lower_inward = dn_at & (D1_px > D0_px)
    upper_non_inward = up_at & ~(U1_px < U0_px)
    lower_non_inward = dn_at & ~(D1_px > D0_px)

    # any inward anywhere in the interval (遍历 change positions 的小切片)
    any_up_inward = np.zeros(S, dtype=bool)
    any_dn_inward = np.zeros(S, dtype=bool)
    up_px_c = U_px[up_changes] if len(up_changes) else np.empty(0)
    dn_px_c = D_px[dn_changes] if len(dn_changes) else np.empty(0)
    lo_u = np.searchsorted(up_changes, t0, side="right")
    hi_u = np.searchsorted(up_changes, hi, side="right")
    lo_d = np.searchsorted(dn_changes, t0, side="right")
    hi_d = np.searchsorted(dn_changes, hi, side="right")
    for s in np.flatnonzero(hi_u > lo_u):
        if np.min(up_px_c[lo_u[s]:hi_u[s]]) < U0_px[s]:
            any_up_inward[s] = True
    for s in np.flatnonzero(hi_d > lo_d):
        if np.max(dn_px_c[lo_d[s]:hi_d[s]]) > D0_px[s]:
            any_dn_inward[s] = True

    n_up_tr = count_in_range(up_changes, t0, hi)
    n_dn_tr = count_in_range(dn_changes, t0, hi)
    n_any_tr = count_in_range(any_changes, t0, hi)

    # activation provenance for first inward change
    prov_up_ok = np.ones(S, dtype=bool)
    prov_dn_ok = np.ones(S, dtype=bool)
    new_up_act = np.full(S, -1, dtype=np.int64)
    new_dn_act = np.full(S, -1, dtype=np.int64)
    new_up_act[upper_inward] = U_act[safe[upper_inward]]
    new_dn_act[lower_inward] = D_act[safe[lower_inward]]
    prov_up_ok[upper_inward] = new_up_act[upper_inward] > t0[upper_inward]
    prov_dn_ok[lower_inward] = new_dn_act[lower_inward] > t0[lower_inward]

    bars_to_first = np.where(reconfigured, first - t0, -1).astype(np.int64)
    frac = np.where(reconfigured & (dur > 0),
                    (first - t0) / np.maximum(dur, 1), np.nan)

    return dict(
        frozen_duration_bars=dur.astype(np.int64),
        reconfigured_before_resolution=reconfigured,
        first_reconfig_bar_index=first,
        bars_to_first_reconfig=bars_to_first,
        episode_fraction_to_first_reconfig=frac,
        first_reconfig_side=side,
        upper_inward=upper_inward,
        lower_inward=lower_inward,
        upper_non_inward=upper_non_inward,
        lower_non_inward=lower_non_inward,
        any_upper_inward=any_up_inward,
        any_lower_inward=any_dn_inward,
        any_inward_reconfig=any_up_inward | any_dn_inward,
        n_upper_pair_changes=n_up_tr,
        n_lower_pair_changes=n_dn_tr,
        n_any_pair_change_bars=n_any_tr,
        new_upper_activation_bar=new_up_act,
        new_lower_activation_bar=new_dn_act,
        provenance_upper_ok=prov_up_ok,
        provenance_lower_ok=prov_dn_ok,
        frozen_upper_price=U0_px,
        frozen_lower_price=D0_px,
        first_upper_price=np.where(reconfigured, U1_px, np.nan),
        first_lower_price=np.where(reconfigured, D1_px, np.nan),
        first_reconfig_close=np.where(reconfigured, close[safe], np.nan),
    )


# ===========================================================================
# main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=None)
    args = ap.parse_args()
    symbols = args.symbols or list(FULL_UNIV)

    t_total = time.perf_counter()
    timing = {}

    t0_ = time.perf_counter()
    frames = []
    missing = [s for s in symbols
               if not (CACHE / f"local0_samples_{s}.parquet").exists()]
    if missing:
        raise SystemExit(
            "STOP_STEP0_PAIR_SEQUENCE_INCOMPLETE: frozen LOCAL-0 sample cache "
            f"missing: {missing}")
    for s in symbols:
        frames.append(pd.read_parquet(CACHE / f"local0_samples_{s}.parquet"))
    cached = pd.concat(frames, ignore_index=True)
    timing["cache_load_seconds"] = round(time.perf_counter() - t0_, 2)

    # ---------------- sample counts / block discipline ----------------
    tb1 = cached[(cached["block"] == TRAIN_BLOCK)
                 & cached["label"].isin(RESOLVED_CODES)]
    tb2 = cached[(cached["block"] == TEST_BLOCK)
                 & cached["label"].isin(RESOLVED_CODES)]
    got = dict(tb1_resolved=int(len(tb1)), tb2_resolved=int(len(tb2)))
    print(f"[SAMPLE] {got}")
    # drift guard 只在全品种运行时生效（单品种 smoke 不适用全样本基线）
    if set(symbols) == set(FULL_UNIV) and got != EXPECT:
        raise SystemExit(
            f"STOP_STEP0_SAMPLE_DRIFT: {got} != {EXPECT}")
    n_tb34 = int(cached["block"].isin(["TB3", "TB4"]).sum())
    print(f"[SAMPLE] TB3/TB4 rows present but excluded = {n_tb34}")

    states = pd.concat([tb1, tb2], ignore_index=True)
    n_excluded = int(len(cached) - len(states))
    print(f"[SAMPLE] excluded (ambiguous / right-censor / TB3 / TB4) = "
          f"{n_excluded}")

    # ---------------- pair sequence ----------------
    t0_ = time.perf_counter()
    master = load_master()
    ms_by_sym = {s: master[master["symbol"] == s].copy() for s in symbols}
    bars_by_sym = {s: load_raw_bars(s) for s in symbols}
    seq_by_sym = {}
    for s in symbols:
        lc = build_corrected_lifecycle(ms_by_sym[s], bars_by_sym[s])
        seq_by_sym[s] = build_bar_pair_arrays(s, ms_by_sym[s],
                                              bars_by_sym[s], lc)
    timing["pair_sequence_prepare_seconds"] = round(time.perf_counter() - t0_, 2)

    # ---------------- parity vs frozen LOCAL-0 pair keys ----------------
    print("[PARITY] full-bar pair sequence vs frozen sample cache")
    mism = 0
    for s in symbols:
        c = cached[cached["symbol"] == s]
        q = seq_by_sym[s]
        bi = c["decision_bar_index"].to_numpy()
        mism += int((q["upper_group"][bi] != c["upper_group"].to_numpy()).sum())
        mism += int((q["lower_group"][bi] != c["lower_group"].to_numpy()).sum())
        mism += int((~np.isclose(q["upper_price"][bi],
                                 c["upper_price"].to_numpy(float))).sum())
        mism += int((~np.isclose(q["lower_price"][bi],
                                 c["lower_price"].to_numpy(float))).sum())
    print(f"[PARITY] mismatches = {mism}")
    if mism:
        raise SystemExit(
            "STOP_STEP0_PAIR_SEQUENCE_INCOMPLETE: recomputed pair sequence "
            f"does not match frozen LOCAL-0 pair keys ({mism} mismatches)")

    # ---------------- audit ----------------
    t0_ = time.perf_counter()
    parts = []
    bar_cmp = 0
    for s in symbols:
        g = states[states["symbol"] == s]
        if len(g) == 0:
            continue
        q = seq_by_sym[s]
        t0i = g["decision_bar_index"].to_numpy()
        T = g["resolution_bar_index"].to_numpy()
        if not bool((T > t0i).all()):
            raise SystemExit(
                "STOP_STEP0_PAIR_SEQUENCE_INCOMPLETE: resolution_bar <= "
                f"decision_bar in {s}")
        if int(T.max()) >= q["n"]:
            raise SystemExit(
                "STOP_STEP0_PAIR_SEQUENCE_INCOMPLETE: resolution beyond bars")
        r = audit_states(q, np.asarray(bars_by_sym[s]["c"], dtype=np.float64),
                         t0i, T)
        r = {k: v for k, v in r.items()}
        r["symbol"] = g["symbol"].to_numpy(object)
        r["decision_bar_index"] = t0i
        r["resolution_bar_index"] = T
        r["frozen_outcome"] = g["label"].to_numpy()
        r["block"] = g["block"].to_numpy(object)
        parts.append(pd.DataFrame(r))
        bar_cmp += int(np.sum(T - t0i))
    audit = pd.concat(parts, ignore_index=True)
    timing["audit_seconds"] = round(time.perf_counter() - t0_, 2)

    audit.to_parquet(CACHE / "step0_state_audit.parquet", index=False)

    n = int(len(audit))
    timing["total_seconds"] = round(time.perf_counter() - t_total, 2)
    timing["states_per_second"] = float(n / max(timing["audit_seconds"], 1e-9))
    timing["approximate_bar_comparisons"] = int(bar_cmp)

    # ---------------- primary audit ----------------
    rec = audit["reconfigured_before_resolution"].to_numpy(bool)
    any_inw = audit["any_inward_reconfig"].to_numpy(bool)
    side = audit["first_reconfig_side"].to_numpy()

    def q(a, p):
        return float(np.nanpercentile(a, p)) if len(a) else float("nan")

    btf = audit.loc[rec, "bars_to_first_reconfig"].to_numpy(float)
    frac = audit.loc[rec, "episode_fraction_to_first_reconfig"].to_numpy(float)
    ntr = audit["n_upper_pair_changes"].to_numpy() \
        + audit["n_lower_pair_changes"].to_numpy()

    types = pd.DataFrame([dict(
        metric=k, count=int(v), rate=float(v) / max(n, 1))
        for k, v in [
            ("NONE", int((side == S_NONE).sum())),
            ("UPPER_ONLY", int((side == S_UPPER).sum())),
            ("LOWER_ONLY", int((side == S_LOWER).sum())),
            ("BOTH", int((side == S_BOTH).sum())),
            ("first_UPPER_INWARD", int(audit["upper_inward"].sum())),
            ("first_LOWER_INWARD", int(audit["lower_inward"].sum())),
            ("first_UPPER_NON_INWARD", int(audit["upper_non_inward"].sum())),
            ("first_LOWER_NON_INWARD", int(audit["lower_non_inward"].sum())),
            ("any_UPPER_INWARD", int(audit["any_upper_inward"].sum())),
            ("any_LOWER_INWARD", int(audit["any_lower_inward"].sum())),
        ]])
    types.to_csv(OUT / "step0_reconfig_types.csv", index=False)
    print("[TYPES]")
    print(types.to_string(index=False))

    # duration bins
    db = pd.cut(audit["frozen_duration_bars"],
                bins=[0, 5, 12, 24, 48, 96, 10 ** 9], labels=DUR_LABELS)
    rows = []
    for lab in DUR_LABELS:
        m = (db == lab).to_numpy()
        if not m.any():
            rows.append(dict(duration_bin=lab, n=0, reconfig_rate=np.nan,
                             inward_reconfig_rate=np.nan,
                             mean_n_pair_changes=np.nan))
            continue
        rows.append(dict(
            duration_bin=lab, n=int(m.sum()),
            reconfig_rate=float(rec[m].mean()),
            inward_reconfig_rate=float(any_inw[m].mean()),
            mean_n_pair_changes=float(ntr[m].mean())))
    pd.DataFrame(rows).to_csv(OUT / "step0_duration_bins.csv", index=False)
    print("[DURATION BINS]")
    print(pd.DataFrame(rows).to_string(index=False))

    # by outcome
    rows = []
    for code, nm in ((UP, "UP"), (DOWN, "DOWN")):
        m = (audit["frozen_outcome"].to_numpy() == code)
        rows.append(dict(
            frozen_outcome=nm, n=int(m.sum()),
            reconfig_rate=float(rec[m].mean()) if m.any() else np.nan,
            upper_inward_rate=float(audit["any_upper_inward"].to_numpy()[m].mean())
            if m.any() else np.nan,
            lower_inward_rate=float(audit["any_lower_inward"].to_numpy()[m].mean())
            if m.any() else np.nan,
            mean_first_reconfig_fraction=float(np.nanmean(frac)) if m.any()
            else np.nan))
    pd.DataFrame(rows).to_csv(OUT / "step0_by_outcome.csv", index=False)

    # by symbol
    rows = []
    for s, g in audit.groupby("symbol"):
        m = g["reconfigured_before_resolution"].to_numpy(bool)
        ai = g["any_inward_reconfig"].to_numpy(bool)
        rows.append(dict(
            symbol=s, n=int(len(g)),
            any_reconfig_rate=float(m.mean()),
            inward_reconfig_rate=float(ai.mean()),
            median_bars_to_first_reconfig=float(np.median(
                g.loc[m, "bars_to_first_reconfig"])) if m.any() else np.nan,
            mean_n_pair_changes=float(
                (g["n_upper_pair_changes"].to_numpy()
                 + g["n_lower_pair_changes"].to_numpy()).mean())))
    bs = pd.DataFrame(rows)
    bs.to_csv(OUT / "step0_by_symbol.csv", index=False)
    print("[BY SYMBOL]")
    print(bs.to_string(index=False))

    # ---------------- provenance ----------------
    up_i = audit["upper_inward"].to_numpy(bool)
    dn_i = audit["lower_inward"].to_numpy(bool)
    prov_fail = int((~audit["provenance_upper_ok"].to_numpy(bool)).sum()
                    + (~audit["provenance_lower_ok"].to_numpy(bool)).sum())
    print(f"[PROVENANCE] first-inward states: upper={int(up_i.sum())} "
          f"lower={int(dn_i.sum())}  failures={prov_fail}")

    # ---------------- examples ----------------
    ex = audit[rec].head(200)[[
        "symbol", "decision_bar_index", "resolution_bar_index",
        "frozen_outcome", "frozen_duration_bars", "first_reconfig_bar_index",
        "bars_to_first_reconfig", "episode_fraction_to_first_reconfig",
        "frozen_upper_price", "first_upper_price",
        "frozen_lower_price", "first_lower_price", "first_reconfig_close",
        "upper_inward", "lower_inward", "new_upper_activation_bar",
        "new_lower_activation_bar"]].copy()
    ex["first_reconfig_side"] = SIDE_NAMES[audit.loc[rec, "first_reconfig_side"]
                                           .to_numpy()[:200]]
    ex.to_csv(OUT / "step0_first_reconfig_examples.csv", index=False)

    non = audit[audit["upper_non_inward"] | audit["lower_non_inward"]]
    non = non.head(200)[[
        "symbol", "decision_bar_index", "resolution_bar_index",
        "frozen_duration_bars", "first_reconfig_bar_index",
        "frozen_upper_price", "first_upper_price",
        "frozen_lower_price", "first_lower_price", "first_reconfig_close",
        "upper_non_inward", "lower_non_inward",
        "new_upper_activation_bar", "new_lower_activation_bar"]]
    non.to_csv(OUT / "step0_non_inward_examples.csv", index=False)

    summary = dict(
        experiment="STEP-0 local step reconfiguration audit",
        question=("After freezing the nearest upper/lower liquidity at "
                  "decision time, how often is the current nearest pair "
                  "reconfigured before the frozen pair is penetrated?"),
        scope=dict(blocks=["TB1", "TB2"], tb3_tb4_excluded=True,
                   tb3_tb4_rows_present=n_tb34,
                   n_excluded_non_resolved=n_excluded),
        sample=dict(expect=EXPECT, got=got, n_states=n),
        primary=dict(
            n_states=n,
            n_any_reconfig=int(rec.sum()),
            rate_any_reconfig=float(rec.mean()),
            n_any_inward_reconfig=int(any_inw.sum()),
            rate_any_inward_reconfig=float(any_inw.mean()),
            first_type_counts={SIDE_NAMES[i]: int((side == i).sum())
                               for i in range(4)},
            first_inward=dict(
                upper=int(audit["upper_inward"].sum()),
                lower=int(audit["lower_inward"].sum()),
                non_inward_upper=int(audit["upper_non_inward"].sum()),
                non_inward_lower=int(audit["lower_non_inward"].sum())),
            bars_to_first_reconfig=dict(
                mean=float(np.mean(btf)) if len(btf) else None,
                p50=q(btf, 50), p90=q(btf, 90), p95=q(btf, 95), p99=q(btf, 99)),
            episode_fraction=dict(
                mean=float(np.nanmean(frac)) if len(frac) else None,
                p50=q(frac, 50), p90=q(frac, 90)),
            pair_transitions=dict(
                mean=float(np.mean(ntr)), p50=q(ntr, 50), p90=q(ntr, 90),
                n_zero=int((ntr == 0).sum()), n_one=int((ntr == 1).sum()),
                n_two=int((ntr == 2).sum()), n_three_plus=int((ntr >= 3).sum())),
        ),
        provenance=dict(
            first_inward_upper=int(up_i.sum()),
            first_inward_lower=int(dn_i.sum()),
            provenance_failures=prov_fail),
        timing=timing,
        causal_note=("STEP-0 is a post-hoc structural audit. It does NOT "
                     "establish predictive value and its future "
                     "reconfiguration fields must not enter any current-step "
                     "prediction model. It only informs how a state episode "
                     "should be segmented."),
    )
    (OUT / "step0_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    if prov_fail > 0:
        bad = audit[(~audit["provenance_upper_ok"].to_numpy(bool))
                    | (~audit["provenance_lower_ok"].to_numpy(bool))]
        bad.head(200).to_csv(OUT / "step0_provenance_failures.csv", index=False)
        (OUT / "step0_summary.json").write_text(
            json.dumps(summary, indent=2, default=str))
        raise SystemExit(
            "STOP_STEP0_INWARD_PROVENANCE_FAIL: "
            f"{prov_fail} first-inward reconfigurations have no identity with "
            "activation_bar > the original decision bar; examples -> "
            f"{OUT / 'step0_provenance_failures.csv'}")

    print("\n[SUMMARY] " + json.dumps(summary["primary"], indent=2,
                                      default=str))
    print(f"[DONE] {timing['total_seconds']}s -> {OUT}")


if __name__ == "__main__":
    main()
