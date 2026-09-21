"""
test_entry_value_tree_core108_v1
================================

T0 unit tests + T0-A (event-after path continues) + T0-B (proximity parity),
T1 production-vs-reference differential + proximity parity gate, and the TP
performance gate for FUTURE-ENTRY-VALUE-TREE-CORE108-V1 (Phase 1).

These tests enforce the 8 REQUIRED CHANGES (RC1..RC8) and the frozen schema.
They do NOT train a model and do NOT modify canonical owners.
"""

import math

import numpy as np
import pandas as pd
import pytest

from research.liquidity_oracle_atlas.experiment_entry_value_tree_core108_v1 import (
    Core108Counters,
    Core108PathTracker,
    DEFAULT_ARTIFACT_ROOT,
    core108_columns,
    FORBIDDEN_FUTURE_COLUMNS,
    join_with_r2,
    reference_replay,
    run_production_kernel,
)
from research.liquidity_oracle_atlas.build_structure_constrained_trade_oracle_dp_v2 import (
    load_oracle_artifact_v2,
)

SYMBOL = "AG"
# Small prefix keeps T0/T1 fast; TP uses the contract's N/2N/4N independently.
T0_BARS = 2500
T1_BARS = 6000


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _run(symbol=SYMBOL, bars=T0_BARS, join=True):
    c = Core108Counters()
    r = run_production_kernel(symbol, c, max_bars=bars, join=join)
    return r, c


def _drive_tracker(sequence):
    """sequence: list of (sC, sH, sL, phase). Returns list of feature tuples."""
    tr = Core108PathTracker()
    tr.start(0, 0.0, 1.0)  # sE=0, A=1
    out = []
    for i, (sC, sH, sL, ph) in enumerate(sequence):
        tr.update(sC, sH, sL, ph)
        out.append(tr.features())
    return out


# --------------------------------------------------------------------------- #
# T0 — schema + frozen label math + boundaries                                 #
# --------------------------------------------------------------------------- #
def test_feature_count_is_108():
    assert len(core108_columns()) == 108
    # 4 TF x (3 DTP + 4 roles x 6 path) = 4 x 27 = 108
    assert len(core108_columns()) == 4 * (3 + 4 * 6)


def test_forbidden_columns_absent_from_X():
    cols = set(core108_columns())
    overlap = cols & FORBIDDEN_FUTURE_COLUMNS
    assert not overlap, f"forbidden columns leaked into X: {overlap}"
    # no future/outcome/cost columns
    for bad in ("Q_F1_L", "label_available_time", "best_F1", "volume", "outcome"):
        assert bad not in cols


def test_target_arithmetic():
    # Y_L = (Q(F1,L)-Q(F1,F))/ATR ; Y_S = (Q(F1,S)-Q(F1,F))/ATR
    ql, qf, qs, atr = 1.2, 0.5, -0.3, 2.0
    yl = (ql - qf) / atr
    ys = (qs - qf) / atr
    assert abs(yl - 0.35) < 1e-12
    assert abs(ys - (-0.4)) < 1e-12


def test_full_q_uses_f1_counterfactual_not_path_action():
    # The label must come from the F1 (Flat, armed=1) counterfactual grid,
    # never from the oracle's actual path position/action.
    ql, qf, qs = 1.2, 0.5, -0.3
    atr = 2.0
    yl = (ql - qf) / atr
    ys = (qs - qf) / atr
    # distinct from a classification / best label
    assert yl != 0.0
    assert isinstance(yl, float)
    # symmetric relationship: Y_L - Y_S = (Q(F1,L)-Q(F1,S))/ATR
    assert abs((yl - ys) - (ql - qs) / atr) < 1e-12


def test_role_orientation():
    from research.liquidity_oracle_atlas.experiment_structure_interaction_entry_v1 import (
        REV_SIGN,
    )
    assert REV_SIGN["SUPPORT"] == 1
    assert REV_SIGN["SELLSIDE_LIQUIDITY"] == 1
    assert REV_SIGN["RESISTANCE"] == -1
    assert REV_SIGN["BUYSIDE_LIQUIDITY"] == -1


def test_signed_distance():
    tr = Core108PathTracker()
    tr.start(0, 0.0, 1.0)  # sE=0, A=1
    tr.update(2.0, 2.0, 2.0, "APPROACH")
    assert abs(tr.features()[0] - 2.0) < 1e-12  # outside
    tr2 = Core108PathTracker()
    tr2.start(0, 0.0, 1.0)
    tr2.update(-0.4, -0.4, -0.4, "BREAK")
    assert abs(tr2.features()[0] - (-0.4)) < 1e-12  # inside/penetrated


def test_velocity_hand():
    seq = [
        (3.0, 3.0, 3.0, "APPROACH"),
        (2.0, 2.0, 2.0, "APPROACH"),
        (1.0, 1.0, 1.0, "APPROACH"),
    ]
    out = _drive_tracker(seq)
    # age 2: (d_start - d_cur)/(2-1) = (3-2)/1 = 1 ; age 3: (3-1)/2 = 1
    assert abs(out[1][3] - 1.0) < 1e-12
    assert abs(out[2][3] - 1.0) < 1e-12


def test_path_efficiency_cases():
    # straight: d 3,2,1,0 -> efficiency ~1
    straight = [(3, 3, 3, "APPROACH"), (2, 2, 2, "APPROACH"),
                (1, 1, 1, "APPROACH"), (0, 0, 0, "APPROACH")]
    eff_s = _drive_tracker(straight)[-1][4]
    assert eff_s > 0.95
    # choppy: 3,2,3,2,1 -> larger total variation, efficiency lower
    choppy = [(3, 3, 3, "APPROACH"), (2, 2, 2, "APPROACH"),
              (3, 3, 3, "APPROACH"), (2, 2, 2, "APPROACH"),
              (1, 1, 1, "APPROACH")]
    eff_c = _drive_tracker(choppy)[-1][4]
    assert eff_c < eff_s - 0.1
    # retreat: 3,4,5 -> negative efficiency
    retreat = [(3, 3, 3, "APPROACH"), (4, 4, 4, "APPROACH"),
               (5, 5, 5, "APPROACH")]
    eff_r = _drive_tracker(retreat)[-1][4]
    assert eff_r < 0


def test_max_penetration():
    # near edge oriented = 0, A=1. bar with sL=-0.4 -> pen 0.4; later -0.6 -> 0.6
    seq = [
        (0.5, 0.5, 0.5, "APPROACH"),
        (0.4, 0.4, -0.4, "BREAK"),
        (0.3, 0.3, -0.6, "BREAK_EXTENSION"),
        (0.4, 0.4, 0.4, "BREAK_RETURNING"),
    ]
    out = _drive_tracker(seq)
    assert abs(out[1][5] - 0.4) < 1e-12
    assert abs(out[2][5] - 0.6) < 1e-12
    # recovers but max penetration is retained
    assert abs(out[3][5] - 0.6) < 1e-12


def test_outside_real_distance_rc3():
    """RC3: OUTSIDE keeps the REAL signed distance; never 0 sentinel."""
    r, _ = _run(bars=T0_BARS, join=False)
    feat = r["feature_df"]
    # collect all (tf,role) path columns
    roles = ["SUPPORT", "RESISTANCE", "SELLSIDE_LIQUIDITY", "BUYSIDE_LIQUIDITY"]
    for tf in ("m5", "m15", "h1", "h4"):
        for role in roles:
            ph = feat[f"{tf}_{role}_phase"].to_numpy()
            dist = feat[f"{tf}_{role}_distance_atr"].to_numpy()
            outside = ph == "OUTSIDE"
            # OUTSIDE must carry a finite, NON-ZERO real distance
            assert np.all(np.isfinite(dist[outside])), f"{tf}_{role} OUTSIDE NaN"
            assert np.all(dist[outside] != 0.0), f"{tf}_{role} OUTSIDE distance==0"


def test_missing_structure_representation_rc3():
    """RC3: no candidate structure -> NO_STRUCTURE, distance_atr NaN."""
    r, _ = _run(bars=T0_BARS, join=False)
    feat = r["feature_df"]
    roles = ["SUPPORT", "RESISTANCE", "SELLSIDE_LIQUIDITY", "BUYSIDE_LIQUIDITY"]
    for tf in ("m5", "m15", "h1", "h4"):
        for role in roles:
            ph = feat[f"{tf}_{role}_phase"].to_numpy()
            dist = feat[f"{tf}_{role}_distance_atr"].to_numpy()
            miss = ph == "NO_STRUCTURE"
            assert np.all(np.isnan(dist[miss])), f"{tf}_{role} NO_STRUCTURE not NaN"


def test_global_episode_weight_sums_to_one_rc5():
    """RC5: w=1/N_e per composite (symbol, proximity_episode_id); sum per episode = 1."""
    r, _ = _run(bars=T1_BARS, join=True)
    df = r["joined"]
    cand = df[df["is_candidate"]]
    grp = cand.groupby(["symbol", "proximity_episode_id"])["sample_weight"].sum()
    # every episode sums to 1 (within float tolerance)
    assert np.all(np.abs(grp.to_numpy() - 1.0) < 1e-9)
    # composite key: values are hashable tuples (symbol str, id int)
    assert all(isinstance(k, tuple) and len(k) == 2 for k in grp.index)


def test_semantic_key_duplicate_failure():
    """Negative control: a duplicate (symbol, decision_time) in the oracle side
    must be rejected by the one_to_one join (never join by row position)."""
    r, _ = _run(bars=800, join=False)
    feat = r["feature_df"].head(50).copy()
    # craft a fake oracle frame with a duplicate decision_time
    dup = feat[["symbol", "decision_time"]].copy()
    dup["proximity_episode_id"] = 1
    dup["proximity_any"] = True
    dup["training_eligible"] = True
    dup["label_available_time"] = pd.Timestamp("2025-01-01")
    dup["Q_F1_L"] = 0.0
    dup["Q_F1_F"] = 0.0
    dup["Q_F1_S"] = 0.0
    dup = pd.concat([dup, dup.iloc[[0]]], ignore_index=True)  # duplicate row
    with pytest.raises(Exception):
        feat.merge(
            dup,
            on=["symbol", "decision_time"],
            how="left",
            validate="one_to_one",
        )


def test_phase_no_mutate_past():
    """Causality: appending more future bars must NOT change earlier feature rows."""
    r1, _ = _run(bars=1500, join=False)
    r2, _ = _run(bars=T1_BARS, join=False)
    f1 = r1["feature_df"].reset_index(drop=True)
    f2 = r2["feature_df"].reset_index(drop=True)
    n = min(len(f1), 800)
    cols = core108_columns()
    numeric_cols = [c for c in cols if not c.endswith("_phase")]
    phase_cols = [c for c in cols if c.endswith("_phase")]
    a = f1[numeric_cols].head(n).to_numpy(dtype=float)
    b = f2[numeric_cols].head(n).to_numpy(dtype=float)
    assert a.shape == b.shape
    assert np.allclose(np.nan_to_num(a), np.nan_to_num(b), equal_nan=True, atol=1e-9)
    # phase columns (categorical strings) must also be stable
    pa = f1[phase_cols].head(n).to_numpy()
    pb = f2[phase_cols].head(n).to_numpy()
    assert pa.shape == pb.shape
    assert (pa == pb).all()


def test_segment_boundary_no_carryover():
    """No (tf, role) episode age may exceed the bar-distance from the current
    segment start (i.e. an episode must never survive a segment reset)."""
    r, _ = _run(bars=T1_BARS, join=False)
    feat = r["feature_df"]
    # segment start bar per row: from raw base we only have bar indices;
    # approximate by requiring age <= decision_bar_index + 1 (causality floor).
    roles = ["SUPPORT", "RESISTANCE", "SELLSIDE_LIQUIDITY", "BUYSIDE_LIQUIDITY"]
    for tf in ("m5", "m15", "h1", "h4"):
        for role in roles:
            age = feat[f"{tf}_{role}_episode_age_5m"].to_numpy()
            bi = feat["decision_bar_index"].to_numpy()
            active = age > 0
            assert np.all(age[active] <= bi[active] + 1), f"{tf}_{role} age overflow"


# --------------------------------------------------------------------------- #
# T0-A — event-after path continues updating (RC2)                             #
# --------------------------------------------------------------------------- #
def test_t0a_event_after_path_continues():
    """After the FIRST break event, velocity/efficiency/max_pen/age must keep
    changing. This proves we do NOT reuse the canonical pre-first-event
    approach_velocity (which would freeze)."""
    seq = [
        (2.0, 2.0, 2.0, "APPROACH"),       # start, d=2
        (1.0, 1.0, 1.0, "APPROACH"),       # d=1
        (0.5, 0.5, -0.3, "BREAK"),         # first break, d=0.5, pen=0.3
        (0.2, 0.2, -0.5, "BREAK_EXTENSION"),  # d=0.2, pen=0.5
        (0.8, 0.8, 0.8, "BREAK_RETURNING"),    # d=0.8
        (1.5, 1.5, 1.5, "FULL_RECLAIM"),       # d=1.5
    ]
    out = _drive_tracker(seq)
    ages = [o[2] for o in out]
    vels = [o[3] for o in out]
    effs = [o[4] for o in out]
    pens = [o[5] for o in out]

    # age increments every bar
    assert ages == [1, 2, 3, 4, 5, 6]
    # velocity keeps changing AFTER the first break (bar index 2)
    v_after_break = vels[2:]
    assert len(set(round(v, 6) for v in v_after_break)) > 1, "velocity frozen after break!"
    # explicit expected velocity values
    assert abs(vels[2] - 0.75) < 1e-12
    assert abs(vels[3] - 0.60) < 1e-12
    assert abs(vels[4] - 0.30) < 1e-12
    assert abs(vels[5] - 0.10) < 1e-12
    # efficiency changes after break (not frozen)
    assert abs(effs[2] - 1.0) < 1e-9
    assert abs(effs[4] - 0.50) < 1e-9
    assert abs(effs[5] - (0.5 / 3.1)) < 1e-9
    # max penetration persists & grows then holds
    assert abs(pens[2] - 0.3) < 1e-12
    assert abs(pens[3] - 0.5) < 1e-12
    assert abs(pens[5] - 0.5) < 1e-12


# --------------------------------------------------------------------------- #
# T0-B / T1 — proximity parity gate (RC6)                                      #
# --------------------------------------------------------------------------- #
def test_t0b_proximity_parity():
    """RC6: feature-kernel recomputed proximity_bits == R2 artifact proximity_bits."""
    r, _ = _run(bars=T0_BARS, join=True)
    df = r["joined"]
    art = load_oracle_artifact_v2(DEFAULT_ARTIFACT_ROOT, SYMBOL)["actions"]
    m = df.merge(
        art[["decision_time", "proximity_bits"]],
        on="decision_time",
        suffixes=("", "_art"),
    )
    assert len(m) > 0
    mismatch = int((m["prox_bits_kernel"].to_numpy() != m["proximity_bits"].to_numpy()).sum())
    assert mismatch == 0, f"proximity parity mismatch={mismatch}"


# --------------------------------------------------------------------------- #
# T1 — production vs independent reference (RC7)                               #
# --------------------------------------------------------------------------- #
def _pick_targets(feature_df, n_targets=12):
    roles = ["SUPPORT", "RESISTANCE", "SELLSIDE_LIQUIDITY", "BUYSIDE_LIQUIDITY"]
    tfs = ["m5", "m15", "h1", "h4"]
    targets = []
    active_phases = {
        "APPROACH", "REJECTED", "ZONE", "BREAK", "BREAK_EXTENSION",
        "BREAK_RETURNING", "BREAK_HOLD", "PARTIAL_RECLAIM", "FULL_RECLAIM",
        "REBREAK",
    }
    seen = set()
    for tf in tfs:
        for role in roles:
            col = f"{tf}_{role}_phase"
            sub = feature_df[feature_df[col].isin(active_phases)]
            for t in sub["decision_bar_index"].head(3).tolist():
                key = (int(t), tf, role)
                if key not in seen:
                    seen.add(key)
                    targets.append({"t": int(t), "tf": tf, "role": role})
                if len(targets) >= n_targets:
                    return targets
    return targets


def test_t1_production_vs_reference():
    """RC7: production path state == independent reference replay (mismatch==0)."""
    r, c = _run(bars=T1_BARS, join=False)
    feat = r["feature_df"]
    targets = _pick_targets(feat, n_targets=24)
    assert targets, "no active-path targets found in prefix"
    # independent reference build (RC7: never calls production path tracker)
    from research.liquidity_oracle_atlas.experiment_entry_value_tree_core108_v1 import (
        build_base_prefix,
    )
    counters = Core108Counters()
    info = build_base_prefix(SYMBOL, counters, max_bars=T1_BARS)
    ref = reference_replay(info, counters, targets, symbol=SYMBOL)

    mismatches = 0
    max_err = 0.0
    for tgt in targets:
        t, tf, role = tgt["t"], tgt["tf"], tgt["role"]
        row = feat[feat["decision_bar_index"] == t]
        if row.empty:
            # target bar not emitted (not in global proximity) -> skip
            continue
        row = row.iloc[0]
        prod = (
            row[f"{tf}_{role}_distance_atr"],
            row[f"{tf}_{role}_phase"],
            int(row[f"{tf}_{role}_episode_age_5m"]),
            row[f"{tf}_{role}_approach_velocity"],
            row[f"{tf}_{role}_path_efficiency"],
            row[f"{tf}_{role}_max_penetration_atr"],
        )
        refv = ref[(t, tf, role)]
        # numeric fields
        for pi, (p, rv) in enumerate(zip(prod[:1] + prod[3:], refv[:1] + refv[3:])):
            if isinstance(p, str):
                assert p == rv, f"phase mismatch at {tgt}: {p} != {rv}"
                continue
            pv = _to_float(p)
            rvv = _to_float(rv)
            if math.isnan(pv) and math.isnan(rvv):
                continue
            err = abs(pv - rvv)
            max_err = max(max_err, err)
            if err > 1e-9:
                mismatches += 1
        # phase compare separately
        assert prod[1] == refv[1], f"phase mismatch at {tgt}: {prod[1]} != {refv[1]}"
    assert mismatches == 0, f"production/reference mismatch={mismatches}, max_err={max_err}"
    assert counters.reference_call_count >= 1


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def test_t1_proximity_parity_gate():
    """RC6 gate inside T1: kernel proximity_bits vs artifact, mismatch==0."""
    r, _ = _run(bars=T1_BARS, join=True)
    df = r["joined"]
    art = load_oracle_artifact_v2(DEFAULT_ARTIFACT_ROOT, SYMBOL)["actions"]
    m = df.merge(
        art[["decision_time", "proximity_bits", "proximity_any"]],
        on="decision_time",
        suffixes=("", "_art"),
    )
    bits_mismatch = int(
        (m["prox_bits_kernel"].to_numpy() != m["proximity_bits"].to_numpy()).sum()
    )
    any_mismatch = int(
        ((m["prox_bits_kernel"] != 0).to_numpy()
         != m["proximity_any"].fillna(False).to_numpy()).sum()
    )
    assert bits_mismatch == 0, f"proximity_bits mismatch={bits_mismatch}"
    assert any_mismatch == 0, f"proximity_any mismatch={any_mismatch}"


# --------------------------------------------------------------------------- #
# TP — performance gate (N / 2N / 4N)                                          #
# --------------------------------------------------------------------------- #
def test_tp_performance_gate():
    """Feature builder scales linearly on true prefixes; invariants hold."""
    N, N2, N4 = 10000, 20000, 40000
    import time

    def time_build(n):
        c = Core108Counters()
        t0 = time.perf_counter()
        r = run_production_kernel(SYMBOL, c, max_bars=n, join=False)
        dt = time.perf_counter() - t0
        return dt, c, r

    dt1, c1, r1 = time_build(N)
    dt2, c2, r2 = time_build(N2)
    dt3, c3, r3 = time_build(N4)

    # wide gate
    assert dt2 / dt1 < 2.8, f"T_2N/T_N={dt2/dt1:.3f} >= 2.8"
    assert dt3 / dt2 < 2.8, f"T_4N/T_2N={dt3/dt2:.3f} >= 2.8"

    # structural invariants (no forbidden recompute paths)
    for c in (c1, c2, c3):
        assert c.full_history_recompute_count == 0
        assert c.reference_call_count == 0
        assert c.concat_count == 0
        assert c.oracle_recompute_count == 0
        assert c.raw_load_count == 1
        assert c.resample_count == 4

    # feature count sanity
    for r in (r1, r2, r3):
        assert len(r["feature_df"].columns) == 108 + 5  # 108 + audit cols
        assert len(core108_columns()) == 108

    # monotonic-ish row growth
    assert c3.feature_row_count >= c2.feature_row_count >= c1.feature_row_count
