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
                          analysis_end: int):
    """forward state machine：每根 bar 最多被访问一次，episodes 不重叠。"""
    ug = seq["upper_group"]
    dg = seq["lower_group"]
    upx = seq["upper_price"]
    dnx = seq["lower_price"]
    n = int(seq["n"])
    disc_idx = np.flatnonzero(disc)

    owner = np.full(n, -1, dtype=np.int64)   # owner[j] for increment (j-1, j]
    episodes = []
    owned = 0
    gap_bars_total = 0
    gap_list = []

    def next_disc_after(i: int) -> int:
        k = np.searchsorted(disc_idx, i, side="right")
        return int(disc_idx[k]) if k < len(disc_idx) else -1

    pending_gap = 0
    cursor = 0
    while cursor <= analysis_end:
        if ug[cursor] < 0 or dg[cursor] < 0:
            cursor += 1
            continue
        s = cursor
        s_up = int(ug[s])
        s_dn = int(dg[s])
        s_upx = float(upx[s])
        s_dnx = float(dnx[s])
        if not (s_upx > close[s] and s_dnx < close[s]):
            cursor += 1
            continue

        limit_disc = next_disc_after(s)      # exclusive end of this segment
        scan_hi = analysis_end if limit_disc < 0 else min(analysis_end,
                                                          limit_disc - 1)
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
            e = scan_hi if scan_hi >= s else s
        if (not censor_end) and (not censor_disc) and e <= s:
            raise SystemExit(
                f"STOP_EPISODE0_ZERO_LENGTH: {sym} start={s} end={e}")

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
            gap_bars_before_episode=int(pending_gap),
            new_upper_group=new_up_g, new_lower_group=new_dn_g,
            new_upper_activation_min_bar=(
                group_min_activation_in(grp, new_up_g, s, e)
                if new_up_g >= 0 else -1),
            new_lower_activation_min_bar=(
                group_min_activation_in(grp, new_dn_g, s, e)
                if new_dn_g >= 0 else -1),
        ))

        # ---- 下一 episode：从 e 开始；若无合法双边 pair 则向前找，记 gap ----
        # 零长度 censored episode（e == s）必须至少前进一根 bar，避免死循环。
        q = e if e > s else e + 1
        while q <= analysis_end and (ug[q] < 0 or dg[q] < 0
                                     or not (upx[q] > close[q]
                                             and dnx[q] < close[q])):
            q += 1
        gap = q - e
        pending_gap = int(gap)
        if gap > 0:
            gap_bars_total += gap
            gap_list.append(int(gap))
        if q > analysis_end:
            break
        if q <= cursor:                      # 安全：cursor 必须严格前进
            q = cursor + 1
            if q > analysis_end:
                break
        cursor = q

    return episodes, owner, owned, gap_bars_total, gap_list


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

    all_eps = []
    all_owner = {}
    gap_lists = {}
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
        eps, owner, owned, gap_tot, gaps = build_episodes_symbol(
            s, seq, grp,
            np.asarray(bars["h"], float), np.asarray(bars["l"], float),
            np.asarray(bars["disc"], bool), np.asarray(bars["c"], float),
            block_of_bar, analysis_end)
        for e in eps:
            e["crosses_tb1_tb2"] = bool(e["start_block"] == TRAIN_BLOCK
                                        and e["end_block"] == TEST_BLOCK)
            e["symbol_analysis_end"] = analysis_end
        all_eps += eps
        all_owner[s] = owner
        gap_lists[s] = (owned, analysis_end, gaps)

    ep = pd.DataFrame(all_eps)
    timing["episode_build_seconds"] = round(time.perf_counter() - t0_, 2)

    n_analysis_bars = int(sum(a for _, a, _ in gap_lists.values()))
    n_owned = int(sum(o for o, _, _ in gap_lists.values()))
    n_gap = int(n_analysis_bars - n_owned)
    # increment 重复占用：builder 内每次写入前 hard assert owner == -1，
    # 因此每根 bar 的 increment owner count 恒为 1。这里再做一次长度一致校验。
    if int(ep["duration_bars"].sum()) != n_owned:
        raise SystemExit(
            "STOP_EPISODE0_OVERLAPPING_INCREMENT: sum(duration) != owned")
    coverage_rate = float(n_owned) / max(n_analysis_bars, 1)

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
    t0_ = time.perf_counter()
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
    gap_col = ep["gap_bars_before_episode"].to_numpy()
    gap_audit = dict(
        n_episodes=n_ep,
        n_end_has_immediate_pair=int((gap_col == 0).sum()),
        n_end_needs_gap=int((gap_col > 0).sum()),
        gap_bars_mean=float(gap_col.mean()),
        gap_bars_p50=float(np.percentile(gap_col, 50)),
        gap_bars_p90=float(np.percentile(gap_col, 90)),
        gap_bars_max=int(gap_col.max()),
        total_gap_bars=int(gap_col.sum()),
        start_pair_equal_end=int(same.sum()),
        one_side_changed=int(one_ch.sum()),
        both_sides_changed=int((up_ch & dn_ch).sum()),
    )
    pd.DataFrame([gap_audit]).to_csv(OUT / "episode0_gap_audit.csv",
                                     index=False)

    timing["audit_seconds"] = round(time.perf_counter() - t0_, 2)
    timing["total_seconds"] = round(time.perf_counter() - t_total, 2)
    timing["bars_processed"] = int(n_analysis_bars)
    timing["bars_per_second"] = float(
        n_analysis_bars / max(timing["episode_build_seconds"], 1e-9))

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
        analysis_window=dict(blocks=["TB1", "TB2"], blocks_source=boundaries,
                             tb3_tb4_read=False),
        n_episodes=n_ep,
        duration=dur_stats,
        same_bar_events=dict(n_single_event=n_single, n_multi_event=n_multi,
                             multi_event_rate=float(n_multi) / max(n_ep, 1)),
        endpoint_family={r["family"]: r for r in fam_rows},
        tb_boundary_audit=tb_audit,
        gap_audit=gap_audit,
        coverage=dict(
            n_analysis_bars=n_analysis_bars,
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
        timing=timing,
        interpretation_limits=(
            "EPISODE-0 only builds a non-overlapping causal structural "
            "episode sequence. It does NOT establish path memory, SMC, "
            "latent state, Semi-Markov structure, tradability or RL."),
    )
    (OUT / "episode0_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    print(f"[EPISODES] n={n_ep}")
    print(f"[DURATION] {dur_stats}")
    print("[ENDPOINT FAMILY]")
    print(pd.DataFrame(fam_rows).to_string(index=False))
    print(f"[SAME-BAR] single={n_single} multi={n_multi} "
          f"rate={n_multi / max(n_ep, 1):.4f}")
    print(f"[COVERAGE] bars={n_analysis_bars} owned={n_owned} gap={n_gap} "
          f"rate={coverage_rate:.4f}")
    print(f"[TB] {tb_audit}")
    print(f"[GAP] {gap_audit}")
    print(f"[KEY] {episode_key_sha256}")
    print(f"[TIMING] {timing}")
    print(f"[DONE] -> {OUT}")


if __name__ == "__main__":
    main()
