"""EPISODE-0 — synthetic / contract tests（A–O + closure P–W）。

运行：
  .venv/bin/python research/liquidity_oracle_atlas/test_episode0_dynamic_episodes_v1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import research.liquidity_oracle_atlas.experiment_episode0_dynamic_episodes_v1 as E  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name} :: {detail}")
        FAILS.append(name)


def mkgrp(groups):
    starts, lengths, price, side, act, exp, pen = [], [], [], [], [], [], []
    for g in groups:
        starts.append(len(act))
        lengths.append(len(g["act"]))
        price.append(g["price"])
        side.append(g["side"])
        act += g["act"]
        exp += g["exp"]
        pen += g["pen"]
    return dict(group_starts=np.array(starts, np.int64),
                group_lengths=np.array(lengths, np.int64),
                price=np.array(price, float),
                side=np.array(side, np.int64),
                act=np.array(act, np.int64),
                exp=np.array(exp, np.int64),
                pen=np.array(pen, np.int64))


def base_grp():
    # 0:frozen upper 105 | 1:frozen lower 95 | 2:new upper 103 | 3:new lower 97
    return mkgrp([
        dict(price=105.0, side=1, act=[0], exp=[20], pen=[-1]),
        dict(price=95.0, side=-1, act=[0], exp=[20], pen=[-1]),
        dict(price=103.0, side=1, act=[3], exp=[20], pen=[-1]),
        dict(price=97.0, side=-1, act=[3], exp=[20], pen=[-1]),
    ])


def mkseq(n, ug, dg, upx, dnx, close):
    return dict(upper_group=np.array(ug, np.int64),
                lower_group=np.array(dg, np.int64),
                upper_price=np.array(upx, float),
                lower_price=np.array(dnx, float),
                close=np.array(close, float), n=n)


def run(seq, grp, high, low, disc, close, ae, block=None):
    n = seq["n"]
    b = np.array(block if block is not None else ["TB1"] * n, dtype=object)
    bet = (np.arange(n, dtype="int64") * 300_000_000_000
           ).astype("datetime64[ns]")
    return E.build_episodes_symbol("AG", seq, grp,
                                   np.asarray(high, float),
                                   np.asarray(low, float),
                                   np.asarray(disc, bool),
                                   np.asarray(close, float), b, bet, ae)


def flat(n):
    return ([0] * n, [1] * n, [105.0] * n, [95.0] * n, [100.0] * n)


# ---------------------------------------------------------------- A / B / C
def test_A_B_C_penetration_bits():
    n = 8
    ug, dg, upx, dnx, close = flat(n)
    highA = [101, 101, 101, 101, 106, 101, 101, 101]
    eps, _, _ = run(mkseq(n, ug, dg, upx, dnx, close), base_grp(),
                    highA, [99] * n, [False] * n, close, 7)
    check("A UP penetration endpoint",
          eps[0]["event_mask"] == E.BIT_UP_PEN and eps[0]["end_bar"] == 4,
          str(eps[:1]))

    eps2, _, _ = run(mkseq(n, ug, dg, upx, dnx, close), base_grp(),
                     [101] * n, [99] * n, [False] * n, close, 3)
    check("A/B/C no event -> CENSOR_ANALYSIS_END",
          eps2[0]["event_mask"] == 0 and eps2[0]["censor_analysis_end"]
          and eps2[0]["end_bar"] == 3, str(eps2[:1]))

    lowB = [99, 99, 99, 99, 99, 94, 99, 99]
    epsB, _, _ = run(mkseq(n, ug, dg, upx, dnx, close), base_grp(),
                     [101] * n, lowB, [False] * n, close, 7)
    check("B DOWN penetration endpoint",
          epsB[0]["event_mask"] == E.BIT_DOWN_PEN and epsB[0]["end_bar"] == 5,
          str(epsB[:1]))

    highC = [101, 106, 101, 101, 101, 101, 101, 101]
    lowC = [99, 94, 99, 99, 99, 99, 99, 99]
    epsC, _, _ = run(mkseq(n, ug, dg, upx, dnx, close), base_grp(),
                     highC, lowC, [False] * n, close, 7)
    check("C both penetrated same bar -> mask 1|2",
          epsC[0]["event_mask"] == (E.BIT_UP_PEN | E.BIT_DOWN_PEN)
          and epsC[0]["end_bar"] == 1,
          f"mask={epsC[0]['event_mask']} e={epsC[0]['end_bar']}")


# ---------------------------------------------------------------- D / E
def test_D_E_new_activation_inward():
    n = 8
    ug = [0, 0, 0, 2, 2, 2, 2, 2]
    seq = mkseq(n, ug, [1] * n,
                [105, 105, 105, 103, 103, 103, 103, 103], [95] * n,
                [100.0] * n)
    eps, _, _ = run(seq, base_grp(), [101] * n, [99] * n, [False] * n,
                    seq["close"], 7)
    check("D genuine new upper inward -> BIT_NEW_UPPER_INWARD",
          eps[0]["event_mask"] == E.BIT_NEW_UPPER_INWARD
          and eps[0]["end_bar"] == 3, str(eps[:1]))
    check("D provenance recorded",
          eps[0]["new_upper_group"] == 2
          and eps[0]["new_upper_activation_min_bar"] == 3, str(eps[0]))

    dg2 = [1, 1, 1, 3, 3, 3, 3, 3]
    seq2 = mkseq(n, [0] * n, dg2, [105] * n,
                 [95, 95, 95, 97, 97, 97, 97, 97], [100.0] * n)
    eps2, _, _ = run(seq2, base_grp(), [101] * n, [99] * n, [False] * n,
                     seq2["close"], 7)
    check("E genuine new lower inward -> BIT_NEW_LOWER_INWARD",
          eps2[0]["event_mask"] == E.BIT_NEW_LOWER_INWARD
          and eps2[0]["end_bar"] == 3, str(eps2[:1]))


# ---------------------------------------------------------------- F
def test_F_combined_mask_same_bar():
    n = 8
    ug = [0, 0, 0, 2, 2, 2, 2, 2]
    seq = mkseq(n, ug, [1] * n,
                [105, 105, 105, 103, 103, 103, 103, 103], [95] * n,
                [100.0] * n)
    high = [101, 101, 101, 106, 101, 101, 101, 101]
    eps, _, _ = run(seq, base_grp(), high, [99] * n, [False] * n,
                    seq["close"], 7)
    check("F penetration + new activation same bar -> mask 1|4",
          eps[0]["event_mask"] == (E.BIT_UP_PEN | E.BIT_NEW_UPPER_INWARD),
          str(eps[0]["event_mask"]))


# ---------------------------------------------------------------- G
def test_G_exact_touch_continues():
    n = 8
    ug, dg, upx, dnx, close = flat(n)
    high = [101, 105, 101, 101, 106, 101, 101, 101]
    eps, _, _ = run(mkseq(n, ug, dg, upx, dnx, close), base_grp(),
                    high, [99] * n, [False] * n, close, 7)
    check("G exact TOUCH does not end episode",
          eps[0]["end_bar"] == 4 and eps[0]["event_mask"] == E.BIT_UP_PEN,
          str(eps[:1]))


# ---------------------------------------------------------------- H / I
def test_H_I_eligibility_not_endpoint():
    n = 8
    close = [103.0, 103.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    ug = [0, 0, 2, 2, 2, 2, 2, 2]
    upx = [105, 105, 103, 103, 103, 103, 103, 103]
    grp = base_grp()
    grp["act"] = np.array([0, 0, 0, 3], np.int64)
    seq = mkseq(n, ug, [1] * n, upx, [95] * n, close)
    eps, _, _ = run(seq, grp, [104, 104, 101, 101, 101, 101, 101, 101],
                    [102, 102, 99, 99, 99, 99, 99, 99], [False] * n, close, 5)
    check("H EQUALITY_ELIGIBILITY does not end episode",
          eps[0]["event_mask"] == 0 and eps[0]["censor_analysis_end"],
          str(eps[:1]))

    close2 = [104.0, 104.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    seq2 = mkseq(n, ug, [1] * n, upx, [95] * n, close2)
    eps2, _, _ = run(seq2, grp, [105, 105, 101, 101, 101, 101, 101, 101],
                     [103, 103, 99, 99, 99, 99, 99, 99], [False] * n, close2, 5)
    check("I ACTIVATION_BAR_ELIGIBILITY does not end episode",
          eps2[0]["event_mask"] == 0 and eps2[0]["censor_analysis_end"],
          str(eps2[:1]))


# ---------------------------------------------------------------- J
def test_J_endpoint_at_eligibility_bar():
    n = 8
    ug = [0, 0, 0, 0, 2, 2, 2, 2]
    seq = mkseq(n, ug, [1] * n,
                [105, 105, 105, 105, 103, 103, 103, 103], [95] * n,
                [100.0] * n)
    grp = base_grp()
    grp["act"] = np.array([0, 0, 2, 3], np.int64)
    eps, _, _ = run(seq, grp, [101] * n, [99] * n, [False] * n,
                    seq["close"], 7)
    check("J endpoint at later eligibility bar (4), not activation bar (2)",
          eps[0]["end_bar"] == 4
          and eps[0]["event_mask"] == E.BIT_NEW_UPPER_INWARD, str(eps[:1]))


# ---------------------------------------------------------------- K / O
def test_K_O_chaining_ownership():
    n = 8
    ug, dg, upx, dnx, close = flat(n)
    high = [101, 101, 106, 101, 101, 101, 101, 101]
    eps, owner, st = run(mkseq(n, ug, dg, upx, dnx, close), base_grp(),
                         high, [99] * n, [False] * n, close, 7)
    check("K episode1 ends at bar2 and next episode starts at bar2",
          eps[0]["end_bar"] == 2 and len(eps) >= 2
          and eps[1]["start_bar"] == 2, str(eps[:2]))
    check("O increment owned once (ep0: 1..2, ep1: 3..)",
          int(owner[1]) == 0 and int(owner[2]) == 0 and int(owner[3]) == 1,
          str(owner[:8].tolist()))
    check("O sum(duration) == owned and no duplicate",
          int(sum(e["duration_bars"] for e in eps)) == st["owned"]
          and int(owner[owner >= 0].size) == st["owned"],
          f"owned={st['owned']}")


# ---------------------------------------------------------------- L
def test_L_gap_before_next_episode():
    n = 10
    ug = [0, 0, -1, -1, 0, 0, 0, 0, 0, 0]
    upx = [105, 105, np.nan, np.nan, 105, 105, 105, 105, 105, 105]
    seq = mkseq(n, ug, [1] * n, upx, [95] * n, [100.0] * n)
    high = [101, 101, 106, 101, 101, 101, 101, 101, 101, 101]
    eps, _, _ = run(seq, base_grp(), high, [99] * n, [False] * n,
                    seq["close"], 9)
    check("L gap recorded before next episode (pair missing for 2 bars)",
          len(eps) >= 2 and eps[0]["end_bar"] == 2
          and eps[1]["gap_bars_before_episode"] == 2
          and eps[1]["start_bar"] == 4,
          str([(e["end_bar"], e["gap_bars_before_episode"])
               for e in eps[:2]]))


# ---------------------------------------------------------------- M / P
def test_M_P_analysis_cutoff_single_terminal():
    n = 10
    ug, dg, upx, dnx, close = flat(n)
    high = [101, 101, 101, 101, 101, 106, 101, 101, 101, 101]
    eps, _, _ = run(mkseq(n, ug, dg, upx, dnx, close), base_grp(),
                    high, [99] * n, [False] * n, close, 3)
    check("M CENSOR_ANALYSIS_END and no read beyond cutoff",
          eps[0]["event_mask"] == 0 and eps[0]["censor_analysis_end"]
          and eps[0]["end_bar"] == 3, str(eps[:1]))
    check("P analysis-end censor produces exactly one terminal episode",
          len(eps) == 1 and eps[0]["duration_bars"] > 0, str(len(eps)))
    check("P/W all durations positive",
          all(e["duration_bars"] > 0 for e in eps), str(eps))


# ---------------------------------------------------------------- Q
def test_Q_cursor_at_analysis_end_no_episode():
    n = 6
    ug = [-1, -1, -1, 0, 0, 0]
    dg = [-1, -1, -1, 1, 1, 1]
    upx = [np.nan, np.nan, np.nan, 105.0, 105.0, 105.0]
    dnx = [np.nan, np.nan, np.nan, 95.0, 95.0, 95.0]
    seq = mkseq(n, ug, dg, upx, dnx, [100.0] * n)
    eps, _, st = run(seq, base_grp(), [101] * n, [99] * n, [False] * n,
                     seq["close"], 3)
    check("Q cursor == analysis_end does not start episode",
          len(eps) == 0, str(eps))
    check("T initial gap accounted (3 unowned increments)",
          st["n_initial_gap"] == 3 and st["n_gap"] == 3, str(st))


# ---------------------------------------------------------------- N / R / S
def test_N_R_S_discontinuity():
    n = 10
    ug, dg, upx, dnx, close = flat(n)
    disc = [False, False, False, True, False, False, False, False, False, False]
    high = [101, 101, 101, 101, 101, 101, 101, 101, 101, 101]
    eps, owner, st = run(mkseq(n, ug, dg, upx, dnx, close), base_grp(),
                         high, [99] * n, disc, close, 8)
    check("N discontinuity before structural event -> CENSOR_DISCONTINUITY",
          eps[0]["event_mask"] == 0 and eps[0]["censor_discontinuity"]
          and eps[0]["end_bar"] == 2, str(eps[:1]))
    check("R discontinuity censor does not restart at d-1",
          len(eps) == 2 and eps[1]["start_bar"] >= 3, str(eps[:2]))
    check("S discontinuity crossing increment unowned",
          int(owner[3]) == -1, str(owner[:8].tolist()))
    check("S crossing increment counted as DISCONTINUITY_GAP",
          st["n_discontinuity_gap"] >= 1, str(st))


# ---------------------------------------------------------------- U / V / W
def test_U_V_W_terminal_gap_and_accounting():
    n = 10
    ug = [0, 0, 0, -1, -1, -1, -1, -1, -1, -1]
    dg = [1] * n
    upx = [105, 105, 105] + [np.nan] * 7
    seq = mkseq(n, ug, dg, upx, [95] * n, [100.0] * n)
    high = [101, 101, 101, 106, 101, 101, 101, 101, 101, 101]
    eps, owner, st = run(seq, base_grp(), high, [99] * n, [False] * n,
                         seq["close"], 8)
    check("U terminal gap accounted",
          len(eps) == 1 and eps[0]["end_bar"] == 3
          and st["n_terminal_gap"] == 5,
          f"eps={len(eps)} term={st['n_terminal_gap']}")
    check("V four gap families sum to all gap increments",
          st["n_initial_gap"] + st["n_inter_episode_gap"]
          + st["n_discontinuity_gap"] + st["n_terminal_gap"] == st["n_gap"],
          str(st))
    check("V owned + gap == analysis increments",
          st["owned"] + st["n_gap"] == st["analysis_end"], str(st))
    check("W all emitted durations positive",
          all(e["duration_bars"] > 0 for e in eps), str(eps))


def main():
    test_A_B_C_penetration_bits()
    test_D_E_new_activation_inward()
    test_F_combined_mask_same_bar()
    test_G_exact_touch_continues()
    test_H_I_eligibility_not_endpoint()
    test_J_endpoint_at_eligibility_bar()
    test_K_O_chaining_ownership()
    test_L_gap_before_next_episode()
    test_M_P_analysis_cutoff_single_terminal()
    test_Q_cursor_at_analysis_end_no_episode()
    test_N_R_S_discontinuity()
    test_U_V_W_terminal_gap_and_accounting()
    print(f"\n==== {len(FAILS)} FAIL ====")
    if FAILS:
        print("FAILED:", FAILS)
        sys.exit(1)
    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
