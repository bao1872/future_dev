"""STEP-0.1 — Local Step Reconfiguration Cause Decomposition

===========================================================================
背景（STEP-0 已确认，本轮不重做）
===========================================================================
STEP-0 结论（只按 decision-state window 解释，不是独立 episode）：

    60.5% 的 frozen decision-state windows 在 frozen boundary resolve 前
    观察到 pair change；58.5% 至少一次 inward pair change。

但 STEP-0 只测到 "pair change"，没有区分成因。本轮把第一次 pair
reconfiguration 的原因拆开。

===========================================================================
本轮分类
===========================================================================
FIRST INWARD CHANGE（新边界更近）
    NEW_ACTIVATION           G1 在 t0 没有任何 active identity，
                             且在 (t0, j] 内有 identity activation
                             —— 真正的新 liquidity 生成
    EQUALITY_ELIGIBILITY     G1 在 t0 已 active，但 price(G1) == close[t0]，
                             因严格 > / < 不是候选；价格移动后成为候选
                             —— price-relative eligibility change
    ACTIVE_ELIGIBLE_ALREADY  G1 在 t0 已 active、已在正确方向、且更 inward
                             —— 理论上必须为 0，非 0 即 P0
    OTHER_INWARD             其他；输出实例并 STOP

FIRST NON-INWARD CHANGE
    TOUCH_ELIGIBILITY        frozen 边界被 exact touch（close[j] == U0_price
                             或 == D0_price），严格 > / < 使其临时不再是候选
    NEW_ACTIVATION_NON_INWARD / EXPIRY / OTHER_NON_INWARD

TRANSITION DECOMPOSITION
    touch/equality-driven transition  vs  structural transition
    （原 STEP-0 的 mean 5.26 是否被 equality toggling 放大）

===========================================================================
冻结边界
===========================================================================
不改 LOCAL-0 lifecycle / activation / penetration / expiry / 严格 > < /
nearest-pair kernel；不改 STEP-0 原始输出。
只用 TB1 / TB2；TB3 / TB4 不参与。不训练模型。
禁止 PATH / Signature / SMC / HMM / PGM / RL。

===========================================================================
输出
===========================================================================
    step01_cause_summary.json
    step01_inward_causes.csv
    step01_non_inward_causes.csv
    step01_transition_decomposition.csv
    step01_overlap_audit.csv
    step01_hazard.csv
    step01_other_examples.csv
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

from research.liquidity_oracle_atlas.run_5m_graph_probability_v1 import (  # noqa: E402
    load_raw_bars,
    load_master,
)
from research.liquidity_oracle_atlas.experiment_local_liquidity_transition_v0 import (  # noqa: E402
    FULL_UNIV, UP, DOWN, RESOLVED_CODES, TRAIN_BLOCK, TEST_BLOCK, OUT, CACHE,
    build_corrected_lifecycle, build_level_groups,
    nearest_active_pair_chunk_v2,
)
from research.liquidity_oracle_atlas.experiment_step0_step_reconfiguration_v1 import (  # noqa: E402
    EXPECT, change_positions, next_after,
)

INWARD_CATS = ["NEW_ACTIVATION", "EQUALITY_ELIGIBILITY",
               "ACTIVE_ELIGIBLE_ALREADY", "OTHER_INWARD"]
NON_INWARD_CATS = ["TOUCH_ELIGIBILITY", "NEW_ACTIVATION_NON_INWARD",
                   "EXPIRY", "OTHER_NON_INWARD"]

HAZ_BINS = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 40), (41, 10 ** 9)]
HAZ_LABELS = ["1", "2", "3-5", "6-10", "11-20", "21-40", "41+"]


# ===========================================================================
# pair sequence (+ group CSR) with npz cache
# ===========================================================================
def build_pair_sequence(sym: str, master_sym: pd.DataFrame, bars: dict,
                        lc: dict):
    info = build_level_groups(master_sym, lc, bars)
    n = int(bars["n"])
    close = np.asarray(bars["c"], dtype=np.float64)
    bar_idx = np.arange(n, dtype=np.int64)

    up_g = np.full(n, -1, dtype=np.int64)
    dn_g = np.full(n, -1, dtype=np.int64)
    up_px = np.full(n, np.nan, dtype=np.float64)
    dn_px = np.full(n, np.nan, dtype=np.float64)
    for lo in range(0, n, 256):
        sl = slice(lo, min(lo + 256, n))
        pair = nearest_active_pair_chunk_v2(bar_idx[sl], close[sl], info)
        up_g[sl] = pair["upper_group"]
        dn_g[sl] = pair["lower_group"]
        up_px[sl] = pair["upper_price"]
        dn_px[sl] = pair["lower_price"]

    # group CSR（按 group 排序后的 identity 级数组）
    order = np.argsort(info["id_group"], kind="stable")
    grp = dict(
        group_starts=info["group_starts"].astype(np.int64),
        group_lengths=info["group_lengths"].astype(np.int64),
        price=info["unique_price"].astype(np.float64),
        side=info["unique_side"].astype(np.int64),
        act=info["activation_bar"][order].astype(np.int64),
        exp=info["expiry_bar"][order].astype(np.int64),
        pen=info["penetration_bar"][order].astype(np.int64),
    )
    seq = dict(upper_group=up_g, lower_group=dn_g, upper_price=up_px,
               lower_price=dn_px, close=close, n=n)
    return seq, grp


def save_seq(sym, seq, grp):
    path = CACHE / f"step01_pair_sequence_{sym}.npz"
    np.savez_compressed(
        path,
        upper_group=seq["upper_group"], lower_group=seq["lower_group"],
        upper_price=seq["upper_price"], lower_price=seq["lower_price"],
        close=seq["close"], n=np.int64(seq["n"]),
        group_starts=grp["group_starts"], group_lengths=grp["group_lengths"],
        price=grp["price"], side=grp["side"], act=grp["act"],
        exp=grp["exp"], pen=grp["pen"])
    return path


def load_seq(sym):
    z = np.load(CACHE / f"step01_pair_sequence_{sym}.npz")
    seq = dict(upper_group=z["upper_group"], lower_group=z["lower_group"],
               upper_price=z["upper_price"], lower_price=z["lower_price"],
               close=z["close"], n=int(z["n"]))
    grp = dict(group_starts=z["group_starts"], group_lengths=z["group_lengths"],
               price=z["price"], side=z["side"], act=z["act"], exp=z["exp"],
               pen=z["pen"])
    return seq, grp


I64MAX = np.iinfo(np.int64).max


def group_active_at(grp, g: int, t: int) -> bool:
    s = int(grp["group_starts"][g])
    e = s + int(grp["group_lengths"][g])
    a = grp["act"][s:e]
    if not len(a):
        return False
    ok = (a <= t) & (t < grp["exp"][s:e]) & (
        (grp["pen"][s:e] < 0) | (t < grp["pen"][s:e]))
    return bool(ok.any())


def group_activation_in(grp, g: int, lo: int, hi: int) -> bool:
    s = int(grp["group_starts"][g])
    e = s + int(grp["group_lengths"][g])
    a = grp["act"][s:e]
    if not len(a):
        return False
    return bool(((a > lo) & (a <= hi)).any())


def classify_inward(grp, t0: int, j: int, g1: int, frozen_px: float,
                    close_arr: np.ndarray, is_upper: bool):
    px = float(grp["price"][g1])
    c0 = float(close_arr[t0])
    if not group_active_at(grp, g1, t0):
        if group_activation_in(grp, g1, t0, j):
            return "NEW_ACTIVATION"
        return "OTHER_INWARD"
    if px == c0:
        return "EQUALITY_ELIGIBILITY"
    if is_upper and (px > c0) and (px < frozen_px):
        return "ACTIVE_ELIGIBLE_ALREADY"
    if (not is_upper) and (px < c0) and (px > frozen_px):
        return "ACTIVE_ELIGIBLE_ALREADY"
    return "OTHER_INWARD"


def classify_non_inward(grp, t0: int, j: int, g_new: int, frozen_px: float,
                        close_arr: np.ndarray):
    """第一次 non-inward pair change 的成因。

    TOUCH_ELIGIBILITY  = frozen 边界被 exact touch（close[j] == frozen price），
                         严格 > / < 使其临时不再是候选
    EXPIRY             = 新 group 在 j 已不 active（理论上 resolved interval 内
                         不应发生）
    NEW_ACTIVATION_NON_INWARD = 新 activation 但方向不是 inward
    """
    if float(close_arr[j]) == float(frozen_px):
        return "TOUCH_ELIGIBILITY"
    if g_new >= 0 and not group_active_at(grp, g_new, j):
        return "EXPIRY"
    if g_new >= 0 and group_activation_in(grp, g_new, t0, j):
        return "NEW_ACTIVATION_NON_INWARD"
    return "OTHER_NON_INWARD"


def transition_flags(arr: np.ndarray, close: np.ndarray,
                     group_price: np.ndarray, nn: int):
    """把 pair sequence 的 transition 拆成 touch/equality-driven 与 structural。

    touch/equality-driven 定义：
        出边 group 在 j 被 exact touch（close[j] == price(prev)），或
        入边 group 在 j-1 被 exact touch（close[j-1] == price(cur)）。
    返回 (cum_all, cum_touch, structural_change_positions)
    """
    cp = change_positions(arr)
    touch = np.zeros(len(cp), dtype=bool)
    if len(cp):
        prev = arr[cp - 1]
        cur = arr[cp]
        ok = prev >= 0
        touch[ok] |= (close[cp[ok]] == group_price[prev[ok]])
        ok2 = cur >= 0
        touch[ok2] |= (close[cp[ok2] - 1] == group_price[cur[ok2]])
    cum_all = np.cumsum(np.isin(np.arange(nn), cp).astype(np.int64))
    tflag = np.zeros(nn, dtype=np.int64)
    tflag[cp] = touch.astype(np.int64)
    cum_touch = np.cumsum(tflag)
    return cum_all, cum_touch, cp[~touch]


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
    audit_path = CACHE / "step0_state_audit.parquet"
    if not audit_path.exists():
        raise SystemExit(
            "STOP_STEP01_CACHE_MISSING: step0_state_audit.parquet not found; "
            "refusing to rebuild frozen STEP-0 state.")
    audit = pd.read_parquet(audit_path)
    audit = audit[audit["symbol"].isin(symbols)].reset_index(drop=True)
    timing["cache_load_seconds"] = round(time.perf_counter() - t0_, 2)

    # ---------------- pair sequence (cache or build once) ----------------
    t0_ = time.perf_counter()
    master = load_master()
    ms_by_sym = {s: master[master["symbol"] == s].copy() for s in symbols}
    bars_by_sym = {s: load_raw_bars(s) for s in symbols}
    seqs, grps = {}, {}
    built = []
    for s in symbols:
        p = CACHE / f"step01_pair_sequence_{s}.npz"
        if p.exists():
            seqs[s], grps[s] = load_seq(s)
        else:
            lc = build_corrected_lifecycle(ms_by_sym[s], bars_by_sym[s])
            seqs[s], grps[s] = build_pair_sequence(s, ms_by_sym[s],
                                                   bars_by_sym[s], lc)
            save_seq(s, seqs[s], grps[s])
            built.append(s)
    timing["pair_sequence_seconds"] = round(time.perf_counter() - t0_, 2)
    print(f"[SEQ] reused={len(symbols)-len(built)} built={len(built)} "
          f"seconds={timing['pair_sequence_seconds']}")

    # ---------------- parity vs frozen LOCAL-0 pair keys ----------------
    mism = 0
    for s in symbols:
        c = pd.read_parquet(CACHE / f"local0_samples_{s}.parquet")
        bi = c["decision_bar_index"].to_numpy()
        q = seqs[s]
        mism += int((q["upper_group"][bi] != c["upper_group"].to_numpy()).sum())
        mism += int((q["lower_group"][bi] != c["lower_group"].to_numpy()).sum())
        mism += int((~np.isclose(q["upper_price"][bi],
                                 c["upper_price"].to_numpy(float))).sum())
        mism += int((~np.isclose(q["lower_price"][bi],
                                 c["lower_price"].to_numpy(float))).sum())
    print(f"[PARITY] mismatches = {mism}")
    if mism:
        raise SystemExit("STOP_STEP01_PARITY_FAIL")

    # ---------------- cause audit ----------------
    t0_ = time.perf_counter()
    inc = {k: 0 for k in INWARD_CATS}
    inc_side = {"upper": {k: 0 for k in INWARD_CATS},
                "lower": {k: 0 for k in INWARD_CATS}}
    nonc = {k: 0 for k in NON_INWARD_CATS}
    other_examples = []

    cause_up = np.full(len(audit), "", dtype=object)
    cause_dn = np.full(len(audit), "", dtype=object)
    cause_non_up = np.full(len(audit), "", dtype=object)
    cause_non_dn = np.full(len(audit), "", dtype=object)

    audit = audit.reset_index(drop=True)
    for s, g in audit.groupby("symbol"):
        grp = grps[s]
        close = seqs[s]["close"]
        up_px_arr = seqs[s]["upper_price"]
        dn_px_arr = seqs[s]["lower_price"]
        idx = g.index.to_numpy()
        t0s = g["decision_bar_index"].to_numpy()
        js = g["first_reconfig_bar_index"].to_numpy()
        up_in = g["upper_inward"].to_numpy(bool)
        dn_in = g["lower_inward"].to_numpy(bool)
        up_non = g["upper_non_inward"].to_numpy(bool)
        dn_non = g["lower_non_inward"].to_numpy(bool)
        f_up = g["frozen_upper_price"].to_numpy(float)
        f_dn = g["frozen_lower_price"].to_numpy(float)
        g1_up = g["first_upper_price"].to_numpy(float)
        g1_dn = g["first_lower_price"].to_numpy(float)
        ug = seqs[s]["upper_group"]
        dg = seqs[s]["lower_group"]
        for r in range(len(g)):
            i = idx[r]
            t0 = int(t0s[r])
            j = int(js[r])
            if j < 0:
                continue
            if up_in[r]:
                gg = int(ug[j])
                cat = classify_inward(grp, t0, j, gg, f_up[r], close, True)
                cause_up[i] = cat
                inc[cat] += 1
                inc_side["upper"][cat] += 1
                if cat == "OTHER_INWARD" and len(other_examples) < 500:
                    other_examples.append(dict(
                        side="upper", symbol=s, decision_bar=t0,
                        reconfig_bar=j, new_group=gg,
                        new_group_price=float(grp["price"][gg]),
                        frozen_price=float(f_up[r]), close_t0=float(close[t0]),
                        close_j=float(close[j]),
                        active_at_t0=group_active_at(grp, gg, t0)))
            if dn_in[r]:
                gg = int(dg[j])
                cat = classify_inward(grp, t0, j, gg, f_dn[r], close, False)
                cause_dn[i] = cat
                inc[cat] += 1
                inc_side["lower"][cat] += 1
                if cat == "OTHER_INWARD" and len(other_examples) < 500:
                    other_examples.append(dict(
                        side="lower", symbol=s, decision_bar=t0,
                        reconfig_bar=j, new_group=gg,
                        new_group_price=float(grp["price"][gg]),
                        frozen_price=float(f_dn[r]), close_t0=float(close[t0]),
                        close_j=float(close[j]),
                        active_at_t0=group_active_at(grp, gg, t0)))
            # non-inward
            if up_non[r]:
                cat = classify_non_inward(grp, t0, j, int(ug[j]), f_up[r],
                                          close)
                cause_non_up[i] = cat
                nonc[cat] += 1
            if dn_non[r]:
                cat = classify_non_inward(grp, t0, j, int(dg[j]), f_dn[r],
                                          close)
                cause_non_dn[i] = cat
                nonc[cat] += 1
    timing["cause_audit_seconds"] = round(time.perf_counter() - t0_, 2)

    audit["inward_cause_upper"] = cause_up
    audit["inward_cause_lower"] = cause_dn
    audit["non_inward_cause_upper"] = cause_non_up
    audit["non_inward_cause_lower"] = cause_non_dn
    audit.to_parquet(CACHE / "step01_state_causes.parquet", index=False)

    n = int(len(audit))
    # ---------------- transition decomposition ----------------
    raw_tr = np.zeros(n, dtype=np.int64)
    touch_tr = np.zeros(n, dtype=np.int64)
    struct_first = np.full(n, -1, dtype=np.int64)
    for s, g in audit.groupby("symbol"):
        seq = seqs[s]
        grp = grps[s]
        close = seq["close"]
        nn = seq["n"]
        idx = g.index.to_numpy()
        t0s = g["decision_bar_index"].to_numpy()
        hi = g["resolution_bar_index"].to_numpy() - 1
        sf_up = np.full(len(g), -1, dtype=np.int64)
        sf_dn = np.full(len(g), -1, dtype=np.int64)

        for arr, gprice, key in ((seq["upper_group"], grp["price"], "up"),
                                 (seq["lower_group"], grp["price"], "dn")):
            cum_all, cum_touch, sp = transition_flags(arr, close, gprice, nn)
            raw_tr[idx] += cum_all[hi] - cum_all[t0s]
            touch_tr[idx] += cum_touch[hi] - cum_touch[t0s]
            f = next_after(sp, t0s)
            f = np.where((f >= 0) & (f <= hi), f, -1)
            if key == "up":
                sf_up = f
            else:
                sf_dn = f
        both = np.where(sf_up >= 0, sf_up, sf_dn)
        m = (sf_up >= 0) & (sf_dn >= 0)
        both[m] = np.minimum(sf_up[m], sf_dn[m])
        struct_first[idx] = both

    struct_tr = raw_tr - touch_tr
    audit["n_raw_transitions"] = raw_tr
    audit["n_touch_transitions"] = touch_tr
    audit["n_structural_transitions"] = struct_tr
    audit["first_structural_reconfig_bar"] = struct_first
    audit.to_parquet(CACHE / "step01_state_causes.parquet", index=False)

    # ---------------- overlap audit ----------------
    key = (audit["symbol"].astype(str) + "|"
           + audit["frozen_upper_price"].astype(str) + "|"
           + audit["frozen_lower_price"].astype(str) + "|"
           + audit["resolution_bar_index"].astype(str))
    vc = key.value_counts()
    overlap = dict(
        n_decision_states=n,
        n_unique_frozen_triplets=int(len(vc)),
        decision_states_per_unique_triplet_mean=float(vc.mean()),
        decision_states_per_unique_triplet_p50=float(np.percentile(vc.to_numpy(), 50)),
        decision_states_per_unique_triplet_p90=float(np.percentile(vc.to_numpy(), 90)),
        decision_states_per_unique_triplet_max=int(vc.max()),
    )
    pd.DataFrame([overlap]).to_csv(OUT / "step01_overlap_audit.csv", index=False)

    # ---------------- hazard (descriptive) ----------------
    dur = audit["frozen_duration_bars"].to_numpy()
    struct_k = np.where(struct_first >= 0,
                        struct_first - audit["decision_bar_index"].to_numpy(),
                        -1)
    rows = []
    for (a, b), lab in zip(HAZ_BINS, HAZ_LABELS):
        at_risk = int(((dur >= a) & ((struct_k < 0) | (struct_k >= a))).sum())
        events = int(((struct_k >= a) & (struct_k <= b)).sum())
        rows.append(dict(hazard_bin=lab, at_risk=at_risk, events=events,
                         hazard=float(events / at_risk) if at_risk else np.nan))
    pd.DataFrame(rows).to_csv(OUT / "step01_hazard.csv", index=False)

    # ---------------- outputs ----------------
    inc_rows = []
    for k in INWARD_CATS:
        inc_rows.append(dict(cause=k, n=inc[k],
                             rate_of_states=float(inc[k]) / max(n, 1),
                             upper=inc_side["upper"][k],
                             lower=inc_side["lower"][k]))
    pd.DataFrame(inc_rows).to_csv(OUT / "step01_inward_causes.csv", index=False)

    non_rows = [dict(cause=k, n=nonc[k],
                     rate_of_states=float(nonc[k]) / max(n, 1))
                for k in NON_INWARD_CATS]
    pd.DataFrame(non_rows).to_csv(OUT / "step01_non_inward_causes.csv",
                                  index=False)

    tr_rows = [dict(metric=k,
                    mean=float(np.mean(v)), p50=float(np.percentile(v, 50)),
                    p90=float(np.percentile(v, 90)),
                    n_zero=int((v == 0).sum()), n_one=int((v == 1).sum()),
                    n_two=int((v == 2).sum()), n_three_plus=int((v >= 3).sum()))
               for k, v in (("raw", raw_tr), ("touch_eligibility", touch_tr),
                            ("structural", struct_tr))]
    pd.DataFrame(tr_rows).to_csv(OUT / "step01_transition_decomposition.csv",
                                 index=False)

    if other_examples:
        pd.DataFrame(other_examples).to_csv(OUT / "step01_other_examples.csv",
                                            index=False)

    timing["total_seconds"] = round(time.perf_counter() - t_total, 2)

    n_new = inc["NEW_ACTIVATION"]
    n_eq = inc["EQUALITY_ELIGIBILITY"]
    n_inv = inc["ACTIVE_ELIGIBLE_ALREADY"]
    n_oth = inc["OTHER_INWARD"]
    states_new = int(((audit["inward_cause_upper"] == "NEW_ACTIVATION")
                      | (audit["inward_cause_lower"] == "NEW_ACTIVATION")).sum())
    states_eq = int(((audit["inward_cause_upper"] == "EQUALITY_ELIGIBILITY")
                     | (audit["inward_cause_lower"] == "EQUALITY_ELIGIBILITY")).sum())
    states_touch = int(((audit["non_inward_cause_upper"] == "TOUCH_ELIGIBILITY")
                        | (audit["non_inward_cause_lower"] == "TOUCH_ELIGIBILITY")).sum())

    summary = dict(
        experiment="STEP-0.1 local step reconfiguration cause decomposition",
        base="abe911a034fbce5d486597d267f70b0b1df2b359 (STEP-0)",
        caveat=("All statistics are DECISION-STATE WINDOW statistics, not "
                "independent non-overlapping episodes."),
        n_decision_states=n,
        inward_causes=inc,
        inward_causes_by_side=inc_side,
        non_inward_causes=nonc,
        rates=dict(
            genuine_new_activation_inward_rate=float(states_new) / max(n, 1),
            equality_eligibility_inward_rate=float(states_eq) / max(n, 1),
            touch_eligibility_rate=float(states_touch) / max(n, 1),
            other_rate=float(n_oth) / max(n, 1),
        ),
        transitions={r["metric"]: {k: r[k] for k in
                                   ("mean", "p50", "p90", "n_zero", "n_one",
                                    "n_two", "n_three_plus")}
                     for r in tr_rows},
        overlap=overlap,
        hazard=rows,
        timing=timing,
        duration_caveat=("Cumulative reconfiguration probability rising with "
                         "frozen duration is NOT Semi-Markov evidence: longer "
                         "windows mechanically provide more exposure."),
    )
    (OUT / "step01_cause_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    print("[INWARD CAUSES]")
    print(pd.DataFrame(inc_rows).to_string(index=False))
    print("[NON-INWARD CAUSES]")
    print(pd.DataFrame(non_rows).to_string(index=False))
    print("[TRANSITIONS]")
    print(pd.DataFrame(tr_rows).to_string(index=False))
    print(f"[OVERLAP] {overlap}")
    print("[HAZARD]")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"[TIMING] {timing}")

    if n_inv > 0:
        raise SystemExit(
            "STOP_STEP01_NEAREST_PAIR_INVARIANT_FAIL: "
            f"{n_inv} inward changes where the new group was already active, "
            "on the correct side and inward at the decision bar.")
    if n_oth > 0:
        raise SystemExit(
            "STOP_STEP01_OTHER_INWARD: "
            f"{n_oth} unexplained inward changes; examples -> "
            f"{OUT / 'step01_other_examples.csv'}")

    print(f"[DONE] {timing['total_seconds']}s -> {OUT}")


if __name__ == "__main__":
    main()
