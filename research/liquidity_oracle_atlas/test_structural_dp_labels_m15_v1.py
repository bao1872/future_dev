"""test_structural_dp_labels_m15_v1
===================================

T0 synthetic cases (A-H) + real-data invariant + reference/production parity
for FUT-M15-STRUCTURAL-DP-LABEL-V1.

Run:  ./.venv/bin/python research/liquidity_oracle_atlas/test_structural_dp_labels_m15_v1.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.build_structural_dp_labels_m15_v1 import (
    build_structural_dp_labels_v1,
    compute_best_entry_gap,
    compute_tp_remaining,
    select_target_for_direction,
    verify_reference_production_parity,
)

SYM = "AG"


# --------------------------------------------------------------------------- #
# T0-A: best entry lies inside candidate region -> gap == 0
# --------------------------------------------------------------------------- #
def test_T0_A_gap_zero():
    gap_pts, gap_atr = compute_best_entry_gap(105.0, [(100.0, 110.0)], atr=1.0)
    assert abs(gap_pts) < 1e-12, gap_pts
    assert abs(gap_atr) < 1e-12, gap_atr
    print("T0-A PASS: best_entry_gap_atr == 0 when entry inside region")


# --------------------------------------------------------------------------- #
# T0-B: best entry outside candidate region by known ATR distance
# --------------------------------------------------------------------------- #
def test_T0_B_gap_known():
    # 5 points below a [100,110] zone, atr = 1.0  -> gap = 5.0
    gap_pts, gap_atr = compute_best_entry_gap(95.0, [(100.0, 110.0)], atr=1.0)
    assert abs(gap_pts - 5.0) < 1e-9, gap_pts
    assert abs(gap_atr - 5.0) < 1e-9, gap_atr
    # above
    gap_pts, gap_atr = compute_best_entry_gap(115.0, [(100.0, 110.0)], atr=2.0)
    assert abs(gap_pts - 5.0) < 1e-9, gap_pts
    assert abs(gap_atr - 2.5) < 1e-9, gap_atr
    print("T0-B PASS: best_entry_gap_atr == known ATR distance")


# --------------------------------------------------------------------------- #
# T0-C: early TP before structural target -> remaining_target_atr > 0
# --------------------------------------------------------------------------- #
def test_T0_C_early_tp():
    out = compute_tp_remaining(
        "LONG", entry_price=100.0, exit_price=105.0, target_price=118.0,
        mfe_signed_points=10.0, atr=1.0,
    )
    assert out["target_reached"] == 0.0
    assert out["tp_price"] == 105.0
    assert out["remaining_target_atr"] > 0.0, out
    assert out["tp_atr"] > 0.0
    # tp must not pass target
    assert out["tp_price"] <= 118.0 + 1e-9
    print(f"T0-C PASS: early TP, remaining_target_atr={out['remaining_target_atr']:.4f} > 0")


# --------------------------------------------------------------------------- #
# T0-D: target reached -> remaining_target_atr == 0 and TP == Target
# --------------------------------------------------------------------------- #
def test_T0_D_target_reached():
    out = compute_tp_remaining(
        "LONG", entry_price=100.0, exit_price=110.0, target_price=118.0,
        mfe_signed_points=20.0, atr=1.0,
    )
    assert out["target_reached"] == 1.0
    assert abs(out["tp_price"] - 118.0) < 1e-12
    assert abs(out["remaining_target_atr"]) < 1e-12
    assert abs(out["tp_atr"] - 18.0) < 1e-9
    print("T0-D PASS: target reached, remaining_target_atr == 0, TP == Target")


# --------------------------------------------------------------------------- #
# T0-E / T0-F: directional target selection (LONG / SHORT)
# --------------------------------------------------------------------------- #
def _make_prev_geom(channels, liq_up, liq_down, atr=1.0):
    empty = ([], [], [], atr)
    return {"m15": (channels, liq_up, liq_down, atr), "h1": empty, "h4": empty}


def test_T0_E_long_target():
    geom = _make_prev_geom(
        channels=[(122.0, 118.0, 1.0)],  # (top, bottom) = zone [118, 122]
        liq_up=[{"left": 1, "level": 125.0, "top": 127.0, "bottom": 123.0,
                 "broken": False, "breach_i": None}],
        liq_down=[],
    )
    fs = {}
    tgt = select_target_for_direction("LONG", geom, close=100.0, seg=0, i=10, sr_first_seen=fs)
    assert tgt is not None
    assert tgt["role"] == "RESISTANCE", tgt["role"]
    assert abs(tgt["near_edge"] - 118.0) < 1e-9
    print("T0-E PASS: LONG target = RESISTANCE above price")


def test_T0_F_short_target():
    geom = _make_prev_geom(
        channels=[(99.0, 95.0, 1.0)],  # (top, bottom) = zone [95, 99]
        liq_up=[],
        liq_down=[{"left": 1, "level": 90.0, "top": 92.0, "bottom": 88.0,
                   "broken": False, "breach_i": None}],
    )
    fs = {}
    tgt = select_target_for_direction("SHORT", geom, close=100.0, seg=0, i=10, sr_first_seen=fs)
    assert tgt is not None
    assert tgt["role"] == "SUPPORT", tgt["role"]
    assert abs(tgt["near_edge"] - 99.0) < 1e-9
    print("T0-F PASS: SHORT target = SUPPORT below price")


# --------------------------------------------------------------------------- #
# T0-G / T0-H: lifecycle negatives on REAL output (oracle guarantees)
# --------------------------------------------------------------------------- #
def test_T0_G_H_lifecycle_negatives(df: pd.DataFrame):
    # one candidate -> one entry
    ep_counts = df["candidate_episode_id"].value_counts()
    assert (ep_counts <= 1).all(), "T0-G FAIL: candidate produced >1 entry"
    # previous exit <= next entry (sorted by entry time already)
    et = df["entry_fill_time"].to_numpy()
    xt = df["exit_fill_time"].to_numpy()
    for i in range(len(df) - 1):
        assert xt[i] <= et[i + 1] + pd.Timedelta(seconds=1), "T0-H FAIL: next entry before prev exit"
    print("T0-G PASS: no second entry per candidate")
    print("T0-H PASS: previous exit <= next entry")


# --------------------------------------------------------------------------- #
# Real-data build + invariants + parity
# --------------------------------------------------------------------------- #
def test_real_build_and_invariants():
    df = build_structural_dp_labels_v1(SYM, emit_assertions=True)
    rep = check_invariants_public(df)
    print(f"real build: {len(df)} labels, invariants={rep}")
    # required columns present
    required = [
        "candidate_episode_id", "direction", "best_entry_gap_atr", "tp_atr",
        "remaining_target_atr", "target_price", "tp_price", "exit_reason",
        "next_candidate_start_time", "label_available_time", "atr_owner",
    ]
    for c in required:
        assert c in df.columns, f"missing column {c}"
    # gap/tp atr must be finite & >=0 where target exists
    has_tgt = df["target_price"].notna()
    assert (df.loc[has_tgt, "best_entry_gap_atr"] >= -1e-9).all()
    assert (df.loc[has_tgt, "tp_atr"] >= -1e-9).all()
    assert (df.loc[has_tgt, "remaining_target_atr"] >= -1e-9).all()
    return df


def check_invariants_public(df):
    from research.liquidity_oracle_atlas.build_structural_dp_labels_m15_v1 import (
        check_structural_invariants,
    )
    return check_structural_invariants(df)


def test_parity():
    res = verify_reference_production_parity(SYM, max_bars=4000)
    print(f"parity: {res}")
    assert res["mismatch_count"] == 0, res
    assert res["max_numerical_error"] < 1e-12, res
    print("PARITY PASS: reference == production on shared slice")


def main():
    test_T0_A_gap_zero()
    test_T0_B_gap_known()
    test_T0_C_early_tp()
    test_T0_D_target_reached()
    test_T0_E_long_target()
    test_T0_F_short_target()
    df = test_real_build_and_invariants()
    test_T0_G_H_lifecycle_negatives(df)
    test_parity()
    print("\nALL STRUCTURAL DP-LABEL TESTS PASS")


if __name__ == "__main__":
    main()
