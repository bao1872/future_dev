"""EPISODE-0 — Non-overlapping Dynamic Liquidity Episodes

===========================================================================
目标（唯一）
===========================================================================
把 CONTINUOUS 5m 市场切成真正 **非重叠** 的 dynamic liquidity episodes：

    s = episode start bar（收盘时存在合法双边 nearest pair）

    structural event（只有 4 个 bit）：
        BIT_UP_PEN            = 1    high[j] > frozen_upper_price
        BIT_DOWN_PEN          = 2    low[j]  < frozen_lower_price
        BIT_NEW_UPPER_INWARD  = 4    genuine new upper activation 且成为
                                     inward 合法 nearest upper
        BIT_NEW_LOWER_INWARD  = 8    对称

    endpoint = 最早的 structural event bar e（event_mask 可同时含多个 bit，
    不猜 intrabar 顺序）

    path increments 属于 (s, e]；下一 episode 的 state 从 e 开始
    （bar e 的收盘价不重复计入两个 episode 的价格增量）

非 structural、**不切 episode** 的：
    TOUCH（strict 不等号）
    EQUALITY_ELIGIBILITY
    ACTIVATION_BAR_ELIGIBILITY
    （后两者来自 STEP-0.1 冻结定义）

===========================================================================
冻结边界
===========================================================================
* 复用 gitignored `step01_pair_sequence_<symbol>.npz`（不重建 lifecycle /
  nearest-pair kernel），并做 frozen LOCAL-0 pair-key parity guard。
* 只用 TB1 + TB2 raw bars 决定 endpoint；analysis cutoff = 最后一个 TB2 bar。
  禁止读 TB3/TB4 bar。
* 不改 LOCAL-0 / LOCAL-0.1 / STEP-0 / STEP-0.1 输出与定义。
* 不训练模型。禁止 PATH features / Path Signature / SMC / HMM / PGM / RL /
  PnL。

===========================================================================
输出
===========================================================================
    episode0_summary.json
    episode0_event_masks.csv
    episode0_duration.csv
    episode0_endpoint_family.csv
    episode0_by_symbol.csv
    episode0_gap_audit.csv
（大型 episode cache / increment owner arrays 存 gitignored cache）
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

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.liquidity_oracle_atlas.run_5m_graph_probability_v1 import (  # noqa: E402
    load_raw_bars,
)
from research.liquidity_oracle_atlas.experiment_local_liquidity_transition_v0 import (  # noqa: E402
    FULL_UNIV, TRAIN_BLOCK, TEST_BLOCK, OUT, CACHE, build_blocks,
)
from research.liquidity_oracle_atlas.experiment_step01_cause_decomposition_v1 import (  # noqa: E402
    load_seq, group_active_at, group_activation_in,
)

BIT_UP_PEN = 1
BIT_DOWN_PEN = 2
BIT_NEW_UPPER_INWARD = 4
BIT_NEW_LOWER_INWARD = 8

PEN_BITS = BIT_UP_PEN | BIT_DOWN_PEN
NEW_BITS = BIT_NEW_UPPER_INWARD | BIT_NEW_LOWER_INWARD

DUR_BINS = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 40), (41, 10 ** 9)]
DUR_LABELS = ["1", "2", "3-5", "6-10", "11-20", "21-40", "41+"]

MASK_TEXT = {
    1: "UP_PEN", 2: "DOWN_PEN", 3: "UP_PEN+DOWN_PEN",
    4: "NEW_UPPER", 8: "NEW_LOWER", 12: "NEW_UPPER+NEW_LOWER",
    5: "UP_PEN+NEW_UPPER", 6: "DOWN_PEN+NEW_UPPER",
    9: "UP_PEN+NEW_LOWER", 10: "DOWN_PEN+NEW_LOWER",
    7: "UP_PEN+DOWN_PEN+NEW_UPPER",
    11: "UP_PEN+DOWN_PEN+NEW_LOWER",
    13: "UP_PEN+NEW_UPPER+NEW_LOWER",
    14: "DOWN_PEN+NEW_UPPER+NEW_LOWER",
    15: "ALL_FOUR",
    0: "CENSOR",
}


def group_min_activation_in(grp, g: int, lo: int, hi: int) -> int:
    s = int(grp["group_starts"][g])
    e = s + int(grp["group_lengths"][g])
    a = grp["act"][s:e]
    if not len(a):
        return -1
    m = (a > lo) & (a <= hi)
    return int(a[m].min()) if m.any() else -1


# ===========================================================================
# forward state machine（可合成测试）
# ===========================================================================
def build_episodes_symbol(sym: str, seq: dict, grp: dict, high: np.ndarray,
                          low: np.ndarray, disc: np.ndarray,
                          close: np.ndarray, block_of_bar: np.ndarray,
                          bar_end_time: np.ndarray, analysis_end: int):
    """forward state machine：每根 bar 最多被访问一次，episodes 不重叠。

    终点 / gap 规则（本轮 closure 修正）：
      * analysis_end censor 后 **直接终止** 该 symbol，不再 chain；
      * start 必须留有未来 increment（cursor < analysis_end）；
      * discontinuity censor 后下一次 search 从 discontinuity bar d 开始，
        increment (d-1, d] 归 DISCONTINUITY_GAP 且必须 unowned；
      * 任何 episode 必须 duration > 0。
    """
    ug = seq["upper_group"]
    dg = seq["lower_group"]
    upx = seq["upper_price"]
    dnx = seq["lower_price"]
    n = int(seq["n"])
    disc_idx = np.flatnonzero(disc)

    owner = np.full(n, -1, dtype=np.int64)   # owner[j] for increment (j-1, j]
    episodes = []
    owned = 0

    def valid_pair(i: int) -> bool:
        return bool(ug[i] >= 0 and dg[i] >= 0
                    and upx[i] > close[i] and dnx[i] < close[i])

    def next_disc_after(i: int) -> int:
        k = np.searchsorted(disc_idx, i, side="right")
        return int(disc_idx[k]) if k < len(disc_idx) else -1

    prev_end = 0          # 上一个 episode 的 end（第一个 episode 前视作 bar 0）
    cursor = 0
    while cursor <= analysis_end:
        # 没有未来 increment 的 bar 不得独立形成 episode
        if cursor >= analysis_end:
            break
        if not valid_pair(cursor):
            cursor += 1
            continue
        s = cursor
        s_up = int(ug[s])
        s_dn = int(dg[s])
        s_upx = float(upx[s])
        s_dnx = float(dnx[s])

        limit_disc = next_disc_after(s)      # exclusive end of this segment
        scan_hi = analysis_end if limit_disc < 0 else min(analysis_end,
                                                          limit_disc - 1)
        if scan_hi < s + 1:
            cursor += 1                      # 不可能产生 duration > 0
            continue

        e = -1
        mask = 0
        new_up_g = -1
        new_dn_g = -1
        censor_end = False
        censor_disc = False
        j = s + 1
        while j <= analysis_end:
            if limit_disc >= 0 and j >= limit_disc:
                censor_disc = True
                e = j - 1
                break
            if j > scan_hi:
                censor_end = True
                e = scan_hi
                break
            m = 0
            if high[j] > s_upx:
                m |= BIT_UP_PEN
            if low[j] < s_dnx:
                m |= BIT_DOWN_PEN
            g_up = int(ug[j])
            if g_up >= 0 and g_up != s_up:
                p = float(upx[j])
                if s_upx > p > close[j]:
                    if (not group_active_at(grp, g_up, s)) and \
                            group_activation_in(grp, g_up, s, j):
                        m |= BIT_NEW_UPPER_INWARD
                        new_up_g = g_up
            g_dn = int(dg[j])
            if g_dn >= 0 and g_dn != s_dn:
                p = float(dnx[j])
                if s_dnx < p < close[j]:
                    if (not group_active_at(grp, g_dn, s)) and \
                            group_activation_in(grp, g_dn, s, j):
                        m |= BIT_NEW_LOWER_INWARD
                        new_dn_g = g_dn
            if m:
                mask = m
                e = j
                break
            j += 1
        if e < 0:
            censor_end = True
            e = scan_hi
        if e <= s:
            raise SystemExit(
                "STOP_EPISODE0_NONPOSITIVE_DURATION: "
                f"{sym} start={s} end={e}")

        for k in range(s + 1, e + 1):
            if owner[k] != -1:
                raise SystemExit(
                    "STOP_EPISODE0_OVERLAPPING_INCREMENT: "
                    f"{sym} bar {k} already owned by episode {owner[k]}")
            owner[k] = len(episodes)
            owned += 1

        episodes.append(dict(
            episode_id=len(episodes), symbol=sym,
            start_bar=s, end_bar=e, duration_bars=int(e - s),
            start_time=bar_end_time[s], end_time=bar_end_time[e],
            start_block=str(block_of_bar[s]), end_block=str(block_of_bar[e]),
            start_close=float(close[s]), end_close=float(close[e]),
            start_upper_group=s_up, start_lower_group=s_dn,
            start_upper_price=s_upx, start_lower_price=s_dnx,
            end_upper_group=int(ug[e]), end_lower_group=int(dg[e]),
            end_upper_price=float(upx[e]), end_lower_price=float(dnx[e]),
            event_mask=int(mask),
            up_penetration=bool(mask & BIT_UP_PEN),
            down_penetration=bool(mask & BIT_DOWN_PEN),
            new_upper_inward=bool(mask & BIT_NEW_UPPER_INWARD),
            new_lower_inward=bool(mask & BIT_NEW_LOWER_INWARD),
            censor_analysis_end=bool(mask == 0 and censor_end),
            censor_discontinuity=bool(mask == 0 and censor_disc),
            gap_bars_before_episode=int(s - prev_end),
            gap_bars_after_episode=0,
            new_upper_group=new_up_g, new_lower_group=new_dn_g,
            new_upper_activation_min_bar=(
                group_min_activation_in(grp, new_up_g, s, e)
                if new_up_g >= 0 else -1),
            new_lower_activation_min_bar=(
                group_min_activation_in(grp, new_dn_g, s, e)
                if new_dn_g >= 0 else -1),
        ))
        prev_end = e

        # ---- analysis-end censor：终止该 symbol，不再 chain ----
        if censor_end:
            episodes[-1]["gap_bars_after_episode"] = int(analysis_end - e)
            break

        # ---- 寻找下一个合法 start ----
        # discontinuity censor：下一次 search 至少从 discontinuity bar d 开始，
        # 因此 increment (e, d] 归 gap（其中 crossing increment 属
        # DISCONTINUITY_GAP）。
        q = int(limit_disc) if censor_disc else e
        while q <= analysis_end and not valid_pair(q):
            q += 1
        if q <= analysis_end:
            episodes[-1]["gap_bars_after_episode"] = int(q - e)
        else:
            episodes[-1]["gap_bars_after_episode"] = int(analysis_end - e)
        if q > analysis_end:
            break
        cursor = q

    # ------------------------------------------------------------------
    # explicit gap accounting（四类必须精确覆盖所有 unowned increments）
    # ------------------------------------------------------------------
    idx = np.arange(1, analysis_end + 1, dtype=np.int64)
    unowned = owner[1:analysis_end + 1] == -1 if analysis_end >= 1 else \
        np.zeros(0, dtype=bool)
    disc_gap = unowned & disc[1:analysis_end + 1] if analysis_end >= 1 else \
        np.zeros(0, dtype=bool)
    rest = unowned & ~disc_gap
    if episodes:
        first_start = int(episodes[0]["start_bar"])
        last_end = int(episodes[-1]["end_bar"])
    else:
        first_start = analysis_end + 1
        last_end = 0
    init_gap = rest & (idx <= first_start)
    term_gap = rest & ~init_gap & (idx > last_end)
    inter_gap = rest & ~init_gap & ~term_gap

    # discontinuity crossing increment 必须 unowned
    for d in disc_idx:
        if 1 <= int(d) <= analysis_end and owner[int(d)] != -1:
            raise SystemExit(
                "STOP_EPISODE0_OVERLAPPING_INCREMENT: discontinuity crossing "
                f"increment {d} owned by episode {owner[int(d)]}")

    stats = dict(
        analysis_end=int(analysis_end),
        owned=int(owned),
        n_gap=int(unowned.sum()),
        n_initial_gap=int(init_gap.sum()),
        n_inter_episode_gap=int(inter_gap.sum()),
        n_discontinuity_gap=int(disc_gap.sum()),
        n_terminal_gap=int(term_gap.sum()),
        n_episodes=len(episodes),
    )
    if (stats["n_initial_gap"] + stats["n_inter_episode_gap"]
            + stats["n_discontinuity_gap"] + stats["n_terminal_gap"]) \
            != stats["n_gap"]:
        raise SystemExit(
            "STOP_EPISODE0_GAP_ACCOUNTING_MISMATCH: "
            f"{stats}")
    return episodes, owner, stats


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

    # ---------------- blocks / analysis window ----------------
    t0_ = time.perf_counter()
    bars_by_sym = {s: load_raw_bars(s) for s in symbols}
    all_days, day_block_code, boundaries = build_blocks(bars_by_sym)
    block_names = np.array([f"TB{i+1}" for i in range(4)], dtype=object)
    timing["cache_load_seconds"] = round(time.perf_counter() - t0_, 2)

    # ---------------- frozen LOCAL-0 pair-key parity guard ----------------
    t_parity = time.perf_counter()
    mism = 0
    for s in symbols:
        c = pd.read_parquet(CACHE / f"local0_samples_{s}.parquet")
        bi = c["decision_bar_index"].to_numpy()
        seq0, _ = load_seq(s)
        mism += int((seq0["upper_group"][bi]
                     != c["upper_group"].to_numpy()).sum())
        mism += int((seq0["lower_group"][bi]
                     != c["lower_group"].to_numpy()).sum())
        mism += int((~np.isclose(seq0["upper_price"][bi],
                                 c["upper_price"].to_numpy(float))).sum())
        mism += int((~np.isclose(seq0["lower_price"][bi],
                                 c["lower_price"].to_numpy(float))).sum())
    print(f"[PARITY] pair-key mismatches = {mism}")
    if mism:
        raise SystemExit("STOP_EPISODE0_PAIR_SEQUENCE_INCOMPATIBLE")
    timing["parity_seconds"] = round(time.perf_counter() - t_parity, 2)

    t_build = time.perf_counter()
    all_eps = []
    all_owner = {}
    sym_stats = {}
    for s in symbols:
        seq, grp = load_seq(s)
        bars = bars_by_sym[s]
        td = np.asarray(bars["td"]).astype("datetime64[D]")
        code = day_block_code[np.searchsorted(all_days, td)]
        block_of_bar = block_names[code]
        ok = np.isin(code, [0, 1])          # TB1 / TB2
        if not ok.any():
            continue
        analysis_end = int(np.flatnonzero(ok)[-1])
        bar_end_time = np.asarray(bars["t"]).astype("datetime64[ns]") \
            + np.timedelta64(5, "m")
        eps, owner, st = build_episodes_symbol(
            s, seq, grp,
            np.asarray(bars["h"], float), np.asarray(bars["l"], float),
            np.asarray(bars["disc"], bool), np.asarray(bars["c"], float),
            block_of_bar, bar_end_time, analysis_end)
        for e in eps:
            e["crosses_tb1_tb2"] = bool(e["start_block"] == TRAIN_BLOCK
                                        and e["end_block"] == TEST_BLOCK)
            e["symbol_analysis_end"] = analysis_end
        # G: CENSOR_ANALYSIS_END 必须是该 symbol 的最后一条 episode
        cens_idx = [i for i, e in enumerate(eps) if e["censor_analysis_end"]]
        if cens_idx and cens_idx[-1] != len(eps) - 1:
            raise SystemExit(
                "STOP_EPISODE0_TERMINAL_CENSOR_NOT_LAST: "
                f"{s} censored_at={cens_idx} n_episodes={len(eps)}")
        # H: CENSOR_DISCONTINUITY 后下一 start 必须 >= discontinuity bar
        d_idx = np.flatnonzero(np.asarray(bars["disc"], bool))
        for i, e in enumerate(eps):
            if e["censor_discontinuity"] and i + 1 < len(eps):
                k = np.searchsorted(d_idx, e["end_bar"], side="right")
                d = int(d_idx[k]) if k < len(d_idx) else -1
                if d >= 0 and eps[i + 1]["start_bar"] < d:
                    raise SystemExit(
                        "STOP_EPISODE0_DISCONTINUITY_CHAINING: "
                        f"{s} next_start={eps[i+1]['start_bar']} d={d}")
        all_eps += eps
        all_owner[s] = owner
        sym_stats[s] = st

    ep = pd.DataFrame(all_eps)
    timing["episode_build_seconds"] = round(time.perf_counter() - t_build, 2)

    # ---------------- hard guards ----------------
    n_analysis_increments = int(sum(v["analysis_end"]
                                    for v in sym_stats.values()))
    n_owned = int(sum(v["owned"] for v in sym_stats.values()))
    n_gap = int(sum(v["n_gap"] for v in sym_stats.values()))
    # D: owned + gap == analysis increments
    if n_owned + n_gap != n_analysis_increments:
        raise SystemExit(
            "STOP_EPISODE0_GAP_ACCOUNTING_MISMATCH: owned+gap != analysis "
            f"({n_owned}+{n_gap}!={n_analysis_increments})")
    # A: 所有 episode duration > 0
    if len(ep) and int((ep["duration_bars"] <= 0).sum()):
        raise SystemExit(
            "STOP_EPISODE0_NONPOSITIVE_DURATION: "
            f"{int((ep['duration_bars'] <= 0).sum())} episodes")
    # C: sum(duration) == owned increments
    if int(ep["duration_bars"].sum()) != n_owned:
        raise SystemExit(
            "STOP_EPISODE0_OVERLAPPING_INCREMENT: sum(duration) != owned")
    # E: 四类 gap 精确加总
    gap_total = dict(
        initial=int(sum(v["n_initial_gap"] for v in sym_stats.values())),
        inter=int(sum(v["n_inter_episode_gap"] for v in sym_stats.values())),
        disc=int(sum(v["n_discontinuity_gap"] for v in sym_stats.values())),
        terminal=int(sum(v["n_terminal_gap"] for v in sym_stats.values())),
    )
    if sum(gap_total.values()) != n_gap:
        raise SystemExit(
            "STOP_EPISODE0_GAP_ACCOUNTING_MISMATCH: "
            f"{gap_total} sum={sum(gap_total.values())} != {n_gap}")
    # F: 没有任何 endpoint 超过 TB2 analysis cutoff
    if len(ep) and int((ep["end_bar"] > ep["symbol_analysis_end"]).sum()):
        raise SystemExit("STOP_EPISODE0_BLOCK_LEAK: endpoint after cutoff")
    coverage_rate = float(n_owned) / max(n_analysis_increments, 1)

    # ---------------- TB boundary audit ----------------
    bad = ep[(~ep["start_block"].isin([TRAIN_BLOCK, TEST_BLOCK]))
             | (~ep["end_block"].isin([TRAIN_BLOCK, TEST_BLOCK]))]
    if len(bad):
        raise SystemExit("STOP_EPISODE0_BLOCK_LEAK: episode outside TB1/TB2")
    tb_audit = dict(
        episodes_start_TB1=int((ep["start_block"] == TRAIN_BLOCK).sum()),
        episodes_start_TB2=int((ep["start_block"] == TEST_BLOCK).sum()),
        episodes_cross_TB1_to_TB2=int(ep["crosses_tb1_tb2"].sum()),
    )

    # ---------------- provenance hard guards ----------------
    for _, r in ep.iterrows():
        if r["new_upper_inward"]:
            if r["new_upper_group"] < 0 or \
                    not (r["start_upper_price"] > r["end_upper_price"]
                         > r["end_close"]) or \
                    r["new_upper_activation_min_bar"] <= r["start_bar"] or \
                    r["new_upper_activation_min_bar"] > r["end_bar"]:
                raise SystemExit(
                    "STOP_EPISODE0_NEW_ACTIVATION_PROVENANCE_FAIL: upper "
                    f"episode_id={r['episode_id']}")
        if r["new_lower_inward"]:
            if r["new_lower_group"] < 0 or \
                    not (r["start_lower_price"] < r["end_lower_price"]
                         < r["end_close"]) or \
                    r["new_lower_activation_min_bar"] <= r["start_bar"] or \
                    r["new_lower_activation_min_bar"] > r["end_bar"]:
                raise SystemExit(
                    "STOP_EPISODE0_NEW_ACTIVATION_PROVENANCE_FAIL: lower "
                    f"episode_id={r['episode_id']}")
        if r["event_mask"] == 0 and not (r["censor_analysis_end"]
                                         or r["censor_discontinuity"]):
            raise SystemExit("STOP_EPISODE0_MASK_EMPTY_WITHOUT_CENSOR")

    ep.to_parquet(CACHE / "episode0_episodes.parquet", index=False)

    # ---------------- analysis / outputs ----------------
    t_audit = time.perf_counter()
    n_ep = int(len(ep))
    dur = ep["duration_bars"].to_numpy()

    ep["event_mask_text"] = ep["event_mask"].map(
        lambda v: MASK_TEXT.get(int(v), f"MASK_{int(v)}"))
    mask_rows = (ep.groupby(["event_mask", "event_mask_text"]).size()
                 .reset_index(name="n"))
    mask_rows["rate"] = mask_rows["n"] / max(n_ep, 1)
    mask_rows = mask_rows.sort_values("n", ascending=False)
    mask_rows.to_csv(OUT / "episode0_event_masks.csv", index=False)

    nbits = np.array([bin(int(m)).count("1") if int(m) else 0
                      for m in ep["event_mask"]])
    n_single = int((nbits == 1).sum())
    n_multi = int((nbits >= 2).sum())

    rows = []
    for (a, b), lab in zip(DUR_BINS, DUR_LABELS):
        m = (dur >= a) & (dur <= b)
        rows.append(dict(duration_bin=lab, n=int(m.sum()),
                         rate=float(m.mean())))
    pd.DataFrame(rows).to_csv(OUT / "episode0_duration.csv", index=False)
    dur_stats = dict(
        n_episodes=n_ep,
        mean=float(dur.mean()), p50=float(np.percentile(dur, 50)),
        p75=float(np.percentile(dur, 75)), p90=float(np.percentile(dur, 90)),
        p95=float(np.percentile(dur, 95)), p99=float(np.percentile(dur, 99)),
        max=int(dur.max()))

    pen_only = ((ep["event_mask"] & PEN_BITS) != 0) & ((ep["event_mask"] & NEW_BITS) == 0)
    new_only = ((ep["event_mask"] & NEW_BITS) != 0) & ((ep["event_mask"] & PEN_BITS) == 0)
    both_fam = ((ep["event_mask"] & PEN_BITS) != 0) & ((ep["event_mask"] & NEW_BITS) != 0)
    cens = ep["event_mask"] == 0
    fam_rows = [
        dict(family="PENETRATION_ONLY", n=int(pen_only.sum()),
             rate=float(pen_only.mean())),
        dict(family="NEW_ACTIVATION_ONLY", n=int(new_only.sum()),
             rate=float(new_only.mean())),
        dict(family="PENETRATION_PLUS_NEW_ACTIVATION", n=int(both_fam.sum()),
             rate=float(both_fam.mean())),
        dict(family="CENSOR", n=int(cens.sum()), rate=float(cens.mean())),
        dict(family="BOTH_PENETRATION_subset", n=int((ep["event_mask"] == 3).sum()),
             rate=float((ep["event_mask"] == 3).mean())),
    ]
    pd.DataFrame(fam_rows).to_csv(OUT / "episode0_endpoint_family.csv",
                                  index=False)

    sym_rows = []
    for s, g in ep.groupby("symbol"):
        sym_rows.append(dict(
            symbol=s, n_episodes=int(len(g)),
            mean_duration=float(g["duration_bars"].mean()),
            p50_duration=float(np.percentile(g["duration_bars"], 50)),
            new_activation_only_rate=float(new_only[g.index].mean()),
            penetration_only_rate=float(pen_only[g.index].mean()),
            censor_rate=float(cens[g.index].mean())))
    pd.DataFrame(sym_rows).to_csv(OUT / "episode0_by_symbol.csv", index=False)

    # start/end pair transition
    same = ((ep["start_upper_group"] == ep["end_upper_group"])
            & (ep["start_lower_group"] == ep["end_lower_group"])).to_numpy()
    up_ch = (ep["start_upper_group"] != ep["end_upper_group"]).to_numpy()
    dn_ch = (ep["start_lower_group"] != ep["end_lower_group"]).to_numpy()
    one_ch = (up_ch ^ dn_ch)
    # ---------------- gap audit（四类精确加总 + episode 间 gap 分布） ----
    inter_gap = []
    sym_gap_rows = []
    for s, g in ep.groupby("symbol"):
        gb = g["gap_bars_before_episode"].to_numpy(np.int64)
        ga = g["gap_bars_after_episode"].to_numpy(np.int64)
        inter_gap += list(gb[1:])
        st = sym_stats[s]
        sym_gap_rows.append(dict(
            symbol=s,
            n_episodes=int(len(g)),
            analysis_increments=st["analysis_end"],
            owned_increments=st["owned"],
            initial_gap_increments=st["n_initial_gap"],
            inter_episode_gap_increments=st["n_inter_episode_gap"],
            discontinuity_gap_increments=st["n_discontinuity_gap"],
            terminal_gap_increments=st["n_terminal_gap"],
            gap_increments_total=st["n_gap"],
            accounting_difference=int(
                st["n_gap"] - st["n_initial_gap"] - st["n_inter_episode_gap"]
                - st["n_discontinuity_gap"] - st["n_terminal_gap"]),
            n_episode_pairs_contiguous=int((gb[1:] == 0).sum()),
            initial_gap_bars=int(gb[0]) if len(gb) else st["analysis_end"],
            terminal_gap_bars=int(ga[-1]) if len(ga) else st["analysis_end"],
        ))
    gap_df = pd.DataFrame(sym_gap_rows)
    total_row = dict(
        symbol="TOTAL", n_episodes=int(gap_df["n_episodes"].sum()),
        analysis_increments=n_analysis_increments,
        owned_increments=n_owned,
        initial_gap_increments=gap_total["initial"],
        inter_episode_gap_increments=gap_total["inter"],
        discontinuity_gap_increments=gap_total["disc"],
        terminal_gap_increments=gap_total["terminal"],
        gap_increments_total=n_gap,
        accounting_difference=int(gap_df["accounting_difference"].sum()),
        n_episode_pairs_contiguous=int(
            gap_df["n_episode_pairs_contiguous"].sum()),
        initial_gap_bars=int(gap_df["initial_gap_bars"].sum()),
        terminal_gap_bars=int(gap_df["terminal_gap_bars"].sum()),
    )
    gap_df = pd.concat([gap_df, pd.DataFrame([total_row])],
                       ignore_index=True)
    gap_df.to_csv(OUT / "episode0_gap_audit.csv", index=False)

    ia = np.array(inter_gap, dtype=np.int64)
    gap_audit = dict(
        n_gap_increments_total=n_gap,
        initial_gap_increments=gap_total["initial"],
        inter_episode_gap_increments=gap_total["inter"],
        discontinuity_gap_increments=gap_total["disc"],
        terminal_gap_increments=gap_total["terminal"],
        accounting_difference=int(n_gap - sum(gap_total.values())),
        n_episode_transitions_immediate=int((ia == 0).sum()) if len(ia) else 0,
        n_episode_transitions_with_gap=int((ia > 0).sum()) if len(ia) else 0,
        inter_episode_gap_mean=float(ia.mean()) if len(ia) else 0.0,
        inter_episode_gap_p50=float(np.percentile(ia, 50)) if len(ia) else 0.0,
        inter_episode_gap_p90=float(np.percentile(ia, 90)) if len(ia) else 0.0,
        inter_episode_gap_p99=float(np.percentile(ia, 99)) if len(ia) else 0.0,
        inter_episode_gap_max=int(ia.max()) if len(ia) else 0,
        n_episode_pairs_contiguous=int((ia == 0).sum()) if len(ia) else 0,
        start_pair_equal_end=int(same.sum()),
        one_side_changed=int(one_ch.sum()),
        both_sides_changed=int((up_ch & dn_ch).sum()),
        note=("inter-episode gap distribution is over episode-to-episode "
              "intervals (bars); the four family counters are increment "
              "counts and are the exact partition of all gap increments"),
    )

    timing["audit_seconds"] = round(time.perf_counter() - t_audit, 2)
    timing["total_seconds"] = round(time.perf_counter() - t_total, 2)
    timing["increments_processed"] = int(n_analysis_increments)
    timing["increments_per_second"] = float(
        n_analysis_increments / max(timing["episode_build_seconds"], 1e-9))

    # ---------------- episode key hash ----------------
    key = (ep["symbol"].astype(str) + "|" + ep["start_bar"].astype(str) + "|"
           + ep["end_bar"].astype(str) + "|" + ep["event_mask"].astype(str))
    h = hashlib.sha256()
    h.update("\n".join(sorted(key.tolist())).encode())
    episode_key_sha256 = h.hexdigest()

    summary = dict(
        experiment="EPISODE-0 non-overlapping dynamic liquidity episodes",
        base="a7b0846054581ab071adf08cdbdc708e04130753",
        structural_contract=dict(
            endpoint="min(strict penetration, genuine inward NEW_ACTIVATION)",
            bits={"BIT_UP_PEN": 1, "BIT_DOWN_PEN": 2,
                  "BIT_NEW_UPPER_INWARD": 4, "BIT_NEW_LOWER_INWARD": 8},
            non_structural_path_events=["TOUCH", "EQUALITY_ELIGIBILITY",
                                        "ACTIVATION_BAR_ELIGIBILITY"],
            same_bar_ties="bitmask, intrabar order never guessed",
        ),
        analysis_window=dict(
            blocks=["TB1", "TB2"], blocks_source=boundaries,
            tb3_tb4_used_for_episode_endpoint=False,
            note=("raw/cache files for all blocks may be loaded, but endpoint "
                  "scan is hard-limited to the last TB2 bar; any endpoint "
                  "beyond the TB2 cutoff raises STOP_EPISODE0_BLOCK_LEAK")),
        n_episodes=n_ep,
        duration=dur_stats,
        same_bar_events=dict(n_single_event=n_single, n_multi_event=n_multi,
                             multi_event_rate=float(n_multi) / max(n_ep, 1)),
        endpoint_family={r["family"]: r for r in fam_rows},
        tb_boundary_audit=tb_audit,
        gap_audit=gap_audit,
        coverage=dict(
            n_analysis_increments=n_analysis_increments,
            n_owned_increments=int(n_owned),
            n_gap_increments=int(n_gap),
            coverage_rate=coverage_rate,
            max_increment_owner_count=1,
            note="owner is written exactly once per increment; overlap "
                 "raises STOP_EPISODE0_OVERLAPPING_INCREMENT"),
        compression=dict(
            old_decision_state_windows=239510, new_episodes=n_ep,
            compression_ratio=float(239510) / max(n_ep, 1),
            note="describes data volume reduction only; it is NOT a sample "
                 "independence ratio"),
        episode_key_sha256=episode_key_sha256,
        closure_guards={
            "A_all_duration_positive": bool(
                n_ep == 0 or int((ep["duration_bars"] <= 0).sum()) == 0),
            "B_max_increment_owner_le_1": True,
            "C_sum_duration_eq_owned": bool(
                int(ep["duration_bars"].sum()) == n_owned),
            "D_owned_plus_gap_eq_analysis": bool(
                n_owned + n_gap == n_analysis_increments),
            "E_four_gap_families_exact": bool(
                sum(gap_total.values()) == n_gap),
            "F_no_endpoint_after_tb2_cutoff": bool(
                n_ep == 0 or int((ep["end_bar"]
                                  > ep["symbol_analysis_end"]).sum()) == 0),
            "G_terminal_censor_is_last": True,
            "H_discontinuity_chaining": True,
        },
        timing=timing,
        interpretation_limits=(
            "EPISODE-0 only builds a non-overlapping causal structural "
            "episode sequence. It does NOT establish path memory, SMC, "
            "latent state, Semi-Markov structure, tradability or RL."),
    )
    summary["EPISODE0_CLOSED"] = bool(
        all(summary["closure_guards"].values()))
    (OUT / "episode0_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(f"[CLOSURE] {summary['closure_guards']}")
    print(f"[CLOSURE] EPISODE0_CLOSED = {summary['EPISODE0_CLOSED']}")

    print(f"[EPISODES] n={n_ep}")
    print(f"[DURATION] {dur_stats}")
    print("[ENDPOINT FAMILY]")
    print(pd.DataFrame(fam_rows).to_string(index=False))
    print(f"[SAME-BAR] single={n_single} multi={n_multi} "
          f"rate={n_multi / max(n_ep, 1):.4f}")
    print(f"[COVERAGE] increments={n_analysis_increments} owned={n_owned} "
          f"gap={n_gap} rate={coverage_rate:.4f}")
    print(f"[TB] {tb_audit}")
    print(f"[GAP] {gap_audit}")
    print(f"[KEY] {episode_key_sha256}")
    print(f"[TIMING] {timing}")
    print(f"[DONE] -> {OUT}")


if __name__ == "__main__":
    main()
