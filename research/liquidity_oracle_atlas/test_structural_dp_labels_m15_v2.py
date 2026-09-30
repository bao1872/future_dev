"""test_structural_dp_labels_m15_v2
====================================

T0 mandatory cases (A-I) + real-data invariants + structural-identity audit
+ INDEPENDENT reference-parity (Reference re-implements the math with simple
explicit loops; it does NOT call the production builder).

Run:  PYTHONPATH=. ./.venv/bin/python \
        research/liquidity_oracle_atlas/test_structural_dp_labels_m15_v2.py
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.build_structural_dp_labels_m15_v2 import (
    _group_structural_events,
    solve_event_exit_v2,
    build_structural_dp_labels_v2,
    check_structural_invariants_v2,
    _structures_in_proximity,
)
from research.liquidity_oracle_atlas.experiment_structure_interaction_entry_v1 import (
    bar_zone_distance,
    select_target,
)
from research.liquidity_oracle_atlas.build_dp_proximity_m15_v1 import (
    build_dp_proximity_m15,
)
from research.liquidity_oracle_atlas.build_execution_environment_m15_v1 import (
    run_environment_m15,
)
from research.liquidity_oracle_atlas.build_trade_oracle_dp_m15_one_entry_proximity_v1 import (
    load_oracle_artifact,
    ARTIFACT_ROOT_DIRNAME as ORACLE_ARTIFACT_ROOT,
)


# --------------------------------------------------------------------------- #
# T0-A / T0-B : structural-event identity (pure grouping, no env)
# --------------------------------------------------------------------------- #
def test_T0_A_same_structure_one_event():
    n = 9
    low = np.zeros(n)
    high = np.zeros(n)
    per_bar = [None] * n
    S1 = ("SR|m15|0|0|122|118|1.0", "SR", 118.0, 122.0)
    for t in [0, 1, 2, 5, 6, 7]:
        per_bar[t] = [S1]
    events, raw_runs, proximity_runs, merges = _group_structural_events(per_bar, low, high, n)
    assert len(events) == 1, f"expected 1 event, got {len(events)}"
    assert events[0]["start_bar"] == 0
    assert events[0]["end_bar"] == n
    assert raw_runs == 6                       # six proximity bars
    assert proximity_runs == 2                 # two proximity runs (gap between)
    assert merges == 1                         # one collapsed same-structure run
    print("T0-A PASS: same SR leaves+returns -> 1 structural event / 1 label")


def test_T0_B_next_different_structure():
    n = 8
    low = np.zeros(n)
    high = np.zeros(n)
    per_bar = [None] * n
    S1 = ("SR|m15|0|0|122|118|1.0", "SR", 118.0, 122.0)
    S2 = ("SR|m15|0|1|200|196|1.0", "SR", 196.0, 200.0)
    for t in [0, 1, 2]:
        per_bar[t] = [S1]
    for t in [3, 4, 5]:
        per_bar[t] = [S2]
    events, _, _, _ = _group_structural_events(per_bar, low, high, n)
    assert len(events) == 2, f"expected 2 events, got {len(events)}"
    assert events[0]["end_bar"] == 3, f"event0 must end at next struct start, got {events[0]['end_bar']}"
    assert events[1]["start_bar"] == 3
    print("T0-B PASS: different structure -> previous closes, new event starts")


# --------------------------------------------------------------------------- #
# T0-C / D / E / F / H : exit DP solver (pure function, no env)
# --------------------------------------------------------------------------- #
def test_T0_C_early_tp():
    opens = np.array([100.0, 101, 102, 103, 104, 105, 106])
    highs = opens + 0.5
    lows = opens - 0.5
    atr = 1.0
    r = solve_event_exit_v2("LONG", 0, 1, 100.0, 6, 110.0, opens, highs, lows, atr)
    assert r["exit_reason"] == "DP_EARLY_EXIT"
    assert r["positive_tp_exists"] is True
    assert r["tp_price"] == 106.0
    assert r["remaining_target_atr"] > 0
    print("T0-C PASS: early TP before target -> remaining>0")


def test_T0_D_target_reached():
    opens = np.array([100.0, 101, 102, 103, 104, 105, 110.2, 111])
    highs = opens + 0.5
    lows = opens - 0.5
    atr = 1.0
    r = solve_event_exit_v2("LONG", 0, 1, 100.0, 7, 110.0, opens, highs, lows, atr)
    assert r["exit_reason"] == "TARGET_TOUCH"
    assert r["positive_tp_exists"] is True
    assert abs(r["tp_price"] - 110.0) < 1e-9
    assert abs(r["remaining_target_atr"]) < 1e-9
    print("T0-D PASS: target reached -> TP==Target, remaining=0, real terminal")


def test_T0_E_early_positive():
    opens = np.array([100.0, 102, 104, 101, 100, 99])
    highs = opens + 0.5
    lows = opens - 0.5
    atr = 1.0
    r = solve_event_exit_v2("LONG", 0, 1, 100.0, 5, 110.0, opens, highs, lows, atr)
    assert r["exit_reason"] == "DP_EARLY_EXIT"
    assert r["tp_price"] == 104.0
    assert r["positive_tp_exists"] is True
    assert r["tp_atr"] > 0 and r["remaining_target_atr"] > 0
    print("T0-E PASS: early positive TP (TP<target, remaining>0)")


def test_T0_F_no_positive_tp():
    opens = np.array([100.0, 99, 98, 97, 96, 95])
    highs = opens + 0.5
    lows = opens - 0.5
    atr = 1.0
    r = solve_event_exit_v2("LONG", 0, 1, 100.0, 5, 110.0, opens, highs, lows, atr)
    assert r["exit_reason"] == "DP_LOSS_EXIT"
    assert r["positive_tp_exists"] is False
    assert np.isnan(r["tp_atr"]), "losing exit must NOT be clamped to 0"
    print("T0-F PASS: no positive TP -> positive_tp_exists=False, tp_atr=NA (not 0)")


def test_T0_H_target_touch_is_mandatory():
    opens = np.array([100.0, 105, 110.3, 108, 107, 106])
    highs = opens + 0.6
    lows = opens - 0.6
    atr = 1.0
    r = solve_event_exit_v2("LONG", 0, 1, 100.0, 5, 110.0, opens, highs, lows, atr)
    assert r["exit_reason"] == "TARGET_TOUCH"
    assert abs(r["remaining_target_atr"]) < 1e-9
    print("T0-H PASS: target reached is a real terminal (no pass-without-exit)")


# --------------------------------------------------------------------------- #
# Real-data build + invariants (T0-G / I lifecycle checks live here)
# --------------------------------------------------------------------------- #
def test_real_build_and_invariants():
    df = build_structural_dp_labels_v2("AG", emit_assertions=True)
    rep = check_structural_invariants_v2(df)
    assert rep.get("hard_fail") is None, f"HARD FAIL: {rep['hard_fail']}"
    for k in ["best_entry_gap_atr_min", "tp_atr_min_finite",
              "remaining_target_atr_min_finite", "zero_remaining_max_tp_target_diff",
              "tp_beyond_target_count", "overlapping_trades",
              "entries_per_structural_event_max", "sequential_min_delta_s"]:
        print(f"   {k} = {rep[k]}")
    assert rep["best_entry_gap_atr_min"] >= -1e-9
    assert rep["tp_beyond_target_count"] == 0
    assert rep["overlapping_trades"] == 0
    assert rep["entries_per_structural_event_max"] == 1
    assert rep["zero_remaining_max_tp_target_diff"] <= 1e-6
    print(f"REAL PASS: {len(df)} valid labels; meta={df.attrs['meta']}")
    return df


def test_structural_identity_audit():
    prox = build_dp_proximity_m15("AG")
    n = len(prox)
    prox_any = np.asarray(prox["dp_proximity_any"].to_numpy(), dtype=bool)
    low = prox["low"].to_numpy(float)
    high = prox["high"].to_numpy(float)
    seg_arr = prox["segment"].to_numpy(np.int64)
    env = run_environment_m15("AG", capture_provenance=False)
    geom = env["geom_by_decision"]
    srfs = {}
    per_bar = [None] * n
    for t in range(n):
        if not prox_any[t]:
            continue
        prev = geom[t - 1] if t >= 1 else None
        per_bar[t] = _structures_in_proximity(
            prev, low[t], high[t], int(seg_arr[t]), t, srfs
        )
    runs = defaultdict(list)
    for t in range(n):
        if per_bar[t]:
            for sid, *_ in per_bar[t]:
                runs[sid].append(t)
    found = None
    for sid, bars in runs.items():
        contiguous = all(bars[i + 1] - bars[i] == 1 for i in range(len(bars) - 1))
        if (not contiguous) and len(bars) >= 3:
            found = (sid, bars)
            break
    assert found is not None, "no split-contact structure found (unlikely)"
    sid, bars = found
    df = build_structural_dp_labels_v2("AG", emit_assertions=False)
    ev_with_sid = df[df["structure_id"] == sid]
    assert len(ev_with_sid) == 1, (
        f"split-contact structure {sid} must map to ONE event, got {len(ev_with_sid)}"
    )
    print(f"IDENTITY AUDIT PASS: {sid} contacted over non-contiguous bars "
          f"{bars[0]}..{bars[-1]} -> exactly 1 structural event "
          f"(event_id={int(ev_with_sid.iloc[0]['event_id'])})")


# --------------------------------------------------------------------------- #
# INDEPENDENT reference kernel (simple explicit loops; does NOT call production)
# --------------------------------------------------------------------------- #
def reference_structural_labels_v2(symbol: str, max_bars: int = None) -> pd.DataFrame:
    prox = build_dp_proximity_m15(symbol, max_bars)
    n = len(prox)
    opens = prox["open"].to_numpy(float)
    highs = prox["high"].to_numpy(float)
    lows = prox["low"].to_numpy(float)
    seg_arr = prox["segment"].to_numpy(np.int64)
    td_arr = pd.to_datetime(prox["trading_day"]).to_numpy()

    env = run_environment_m15(symbol, max_bars, capture_provenance=False)
    atr_series = np.asarray(env["features"]["m15_atr"].to_numpy(float))
    geom = env["geom_by_decision"]

    oracle = load_oracle_artifact(os.path.join("artifacts", ORACLE_ARTIFACT_ROOT), symbol)
    assert oracle["ok"]
    trades = oracle["trades"]

    # --- independent per-bar structure recovery ---
    srfs = {}
    per_bar = [None] * n
    for t in range(n):
        prev = geom[t - 1] if t >= 1 else None
        if prev is None:
            continue
        structs = []
        for tf, g in prev.items():
            channels, liq_up, liq_down, atr_tf = g
            if not (np.isfinite(atr_tf) and atr_tf > 0):
                continue
            radius = 0.5 * float(atr_tf)
            for top, bottom, strength in channels:
                if bar_zone_distance(float(lows[t]), float(highs[t]), bottom, top) <= radius:
                    key = (tf, int(seg_arr[t]), round(top, 6), round(bottom, 6), round(strength, 4))
                    if key not in srfs:
                        srfs[key] = t
                    structs.append((f"SR|{tf}|{int(seg_arr[t])}|{srfs[key]}|{top}|{bottom}|{strength}", "SR", float(bottom), float(top)))
            for z in liq_up:
                if z.get("broken"):
                    continue
                if bar_zone_distance(float(lows[t]), float(highs[t]), z["bottom"], z["top"]) <= radius:
                    structs.append((f"LIQ|{tf}|BUYSIDE_LIQUIDITY|{int(seg_arr[t])}|{z['left']}|{z['level']}", "LIQ", float(z["bottom"]), float(z["top"])))
            for z in liq_down:
                if z.get("broken"):
                    continue
                if bar_zone_distance(float(lows[t]), float(highs[t]), z["bottom"], z["top"]) <= radius:
                    structs.append((f"LIQ|{tf}|SELLSIDE_LIQUIDITY|{int(seg_arr[t])}|{z['left']}|{z['level']}", "LIQ", float(z["bottom"]), float(z["top"])))
        per_bar[t] = structs

    # --- independent grouping ---
    events = []
    cur = None
    for t in range(n):
        structs = per_bar[t]
        if not structs:
            continue
        best = None
        bd = None
        for sid, typ, b, tp in structs:
            d = bar_zone_distance(float(lows[t]), float(highs[t]), b, tp)
            if bd is None or d < bd:
                bd = d
                best = (sid, typ, b, tp)
        sid = best[0]
        if cur is None:
            cur = dict(event_id=len(events), structure_id=sid, structure_type=best[1],
                       timeframe=sid.split("|")[1], zone_bottom=best[2], zone_top=best[3],
                       start_bar=t, end_bar=n)
        else:
            if cur["structure_id"] in {s[0] for s in structs}:
                continue
            cur["end_bar"] = t
            events.append(cur)
            cur = dict(event_id=len(events), structure_id=sid, structure_type=best[1],
                       timeframe=sid.split("|")[1], zone_bottom=best[2], zone_top=best[3],
                       start_bar=t, end_bar=n)
    if cur is not None:
        events.append(cur)

    unit_starts = []
    for i in range(n):
        if i == 0 or td_arr[i] != td_arr[i - 1] or seg_arr[i] != seg_arr[i - 1]:
            unit_starts.append(i)
    unit_starts = np.array(unit_starts)

    def next_unit_after(bar):
        later = unit_starts[unit_starts > bar]
        return int(later[0]) if len(later) else None

    rows = []
    for e in events:
        s_bar, e_bar = e["start_bar"], e["end_bar"]
        mask = (trades["entry_fill_index"].to_numpy(int) >= s_bar) & \
               (trades["entry_fill_index"].to_numpy(int) < e_bar)
        sub = trades[mask].sort_values("entry_fill_index")
        if sub.empty:
            continue
        ent = sub.iloc[0]
        direction = ent["direction"]
        edi = int(ent["entry_decision_index"])
        efi = int(ent["entry_fill_index"])
        ep = float(ent["entry_fill_price"])
        atr_v = float(atr_series[edi])
        zb, zt = e["zone_bottom"], e["zone_top"]
        gap = bar_zone_distance(ep, ep, zb, zt)
        gp = geom[edi - 1] if edi >= 1 else None
        roles = ["RESISTANCE", "BUYSIDE_LIQUIDITY"] if direction == "LONG" else ["SUPPORT", "SELLSIDE_LIQUIDITY"]
        cands = []
        for tf, g in (gp or {}).items():
            channels, liq_up, liq_down, atr_tf = g
            for role in roles:
                c = select_target(role, channels, liq_up, liq_down, ep, float(atr_tf), tf, int(seg_arr[edi]), edi, srfs)
                if c is None:
                    continue
                c["tf"] = tf
                cands.append(c)
        tgt = min(cands, key=lambda x: abs(float(x["near_edge"]) - ep)) if cands else None
        target_price = float(tgt["near_edge"]) if tgt else None

        seg_end = e_bar - 1
        nus = next_unit_after(efi)
        if nus is not None:
            seg_end = min(seg_end, nus - 1)
        seg_end = min(seg_end, n - 1)

        s = 1.0 if direction == "LONG" else -1.0
        reached = None
        if target_price is not None:
            tp = float(target_price)
            for t in range(efi, int(seg_end) + 1):
                if direction == "LONG" and highs[t] >= tp:
                    reached = t
                    break
                if direction == "SHORT" and lows[t] <= tp:
                    reached = t
                    break
        if reached is not None:
            tp_price = float(target_price)
            tp_idx = reached
            tp_atr = ((tp_price - ep) if s > 0 else (ep - tp_price)) / atr_v
            positive = True
            remaining = 0.0
            reason = "TARGET_TOUCH"
        else:
            seg_lo = efi + 1
            seg_hi = int(seg_end)
            if seg_hi < seg_lo:
                best_k = efi
                best_pnl = 0.0
            else:
                best_k = seg_lo
                best_pnl = s * (opens[seg_lo] - ep)
                for k in range(seg_lo + 1, seg_hi + 1):
                    p = s * (opens[k] - ep)
                    if p > best_pnl:
                        best_pnl = p
                        best_k = k
            tp_price = float(opens[best_k])
            tp_idx = best_k
            if best_pnl > 1e-9:
                positive = True
                tp_atr = ((tp_price - ep) if s > 0 else (ep - tp_price)) / atr_v
                reason = "DP_EARLY_EXIT"
            else:
                positive = False
                tp_atr = float("nan")
                reason = "DP_LOSS_EXIT"
            remaining = ((target_price - tp_price) if s > 0 else (tp_price - target_price)) / atr_v if target_price is not None else float("nan")

        rows.append({
            "label_id": f"{symbol}_{e['structure_id']}_{efi}",
            "event_id": int(e["event_id"]),
            "structure_id": e["structure_id"],
            "direction": direction,
            "entry_fill_index": efi,
            "entry_fill_price": ep,
            "best_entry_gap_atr": gap / atr_v,
            "tp_price": tp_price,
            "tp_atr": float(tp_atr),
            "positive_tp_exists": bool(positive),
            "target_price": target_price,
            "target_distance_atr": ((target_price - ep) if s > 0 else (ep - target_price)) / atr_v if target_price is not None else float("nan"),
            "remaining_target_atr": float(remaining),
            "exit_reason": reason,
            "exit_fill_index": int(tp_idx),
        })
    return pd.DataFrame(rows)


def test_independent_reference_parity():
    prod = build_structural_dp_labels_v2("AG", emit_assertions=False)
    ref = reference_structural_labels_v2("AG")
    cols = ["best_entry_gap_atr", "tp_atr", "positive_tp_exists", "target_distance_atr",
            "remaining_target_atr", "direction", "exit_reason"]
    m = prod.merge(ref, on="label_id", suffixes=("", "_r"), how="inner")
    assert len(m) > 0, "no shared labels"
    mismatch = 0
    max_err = 0.0
    first_mismatch = None
    for c in cols:
        a = m[c].to_numpy()
        b = m[f"{c}_r"].to_numpy()
        for i in range(len(a)):
            av, bv = a[i], b[i]
            if isinstance(av, float) and (np.isnan(av) or np.isnan(bv)):
                if np.isnan(av) != np.isnan(bv):
                    mismatch += 1
                    if first_mismatch is None:
                        first_mismatch = (m["label_id"].iloc[i], c, av, bv)
                continue
            if av != bv:
                try:
                    err = abs(float(av) - float(bv))
                except Exception:
                    err = 1.0
                if err > 1e-9:
                    mismatch += 1
                    max_err = max(max_err, err)
                    if first_mismatch is None:
                        first_mismatch = (m["label_id"].iloc[i], c, av, bv)
    print(f"REFERENCE PARITY: shared={len(m)} mismatch={mismatch} "
          f"max_err={max_err:.3e} first={first_mismatch}")
    assert mismatch == 0, f"reference/production mismatch: {first_mismatch}"
    print("REFERENCE PARITY PASS: independent kernel == production (mismatch=0)")


if __name__ == "__main__":
    test_T0_A_same_structure_one_event()
    test_T0_B_next_different_structure()
    test_T0_C_early_tp()
    test_T0_D_target_reached()
    test_T0_E_early_positive()
    test_T0_F_no_positive_tp()
    test_T0_H_target_touch_is_mandatory()
    test_real_build_and_invariants()
    test_structural_identity_audit()
    test_independent_reference_parity()
    print("\nALL V2 TESTS PASSED")
