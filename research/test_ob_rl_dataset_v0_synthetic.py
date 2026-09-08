#!/usr/bin/env python3

"""Synthetic Gate-A test for the OB RL Dataset V0.

Runs the REAL builder on synthetic inputs (loaders redirected), and
asserts the semantics that matter:

    Event          touch behaviour / first-retest
    SMC            forward-backward rotation for bull AND short
                   on internal / swing / active OB
    DSA            follow vs fade strictly opposite
    Momentum       canonical string preserved; rel strictly opposite
    RR             target_fit == forward / target_R
    Reward         Phase-1 simulator reused; target / stop /
                   timeout / both-hit conservative / non-contiguous
    Cardinality    N state rows, 7N action rows
    Causality      entry-next-open in the state must STOP
    4h             quarantined

NO real AG/CU/RB/M data is read. Run with:

    python research/test_ob_rl_dataset_v0_synthetic.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import build_ob_rl_dataset_v0 as BM  # noqa: E402
from research import ob_rl_dataset_v0_spec as SP  # noqa: E402
from research.analyze_ob_candidate_v3_phase1 import (  # noqa: E402
    EXIT_NONCONTIG,
    EXIT_GAP_STOP,
    EXIT_GAP_TARGET,
    EXIT_BOTH,
    EXIT_STOP,
    EXIT_TARGET,
    EXIT_TIMEOUT,
)

RNG = np.random.default_rng(20260908)
TMP = Path(tempfile.mkdtemp(prefix="obrl_synth_"))

RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, ok: bool, detail: object = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    if not ok:
        print(f"FAIL  {name} :: {detail}", flush=True)


# ============================================================
# 1) Event semantics
# ============================================================
ev = pd.DataFrame(
    {
        "touch_close_beyond_far_edge": [True, False, False, False],
        "touch_intrabar_far_edge_breach": [True, True, False, False],
    }
)
tb = BM.touch_behavior(ev)
check(
    "touch_behavior 3 classes + precedence",
    list(tb)
    == ["CLOSE_BEYOND", "BREACH_RECLAIM", "NO_BREACH", "NO_BREACH"],
    list(tb),
)

check(
    "quant_state LOW/MID/HIGH/UNKNOWN",
    [
        BM.quant_state(0.10),
        BM.quant_state(0.50),
        BM.quant_state(0.85),
        BM.quant_state(np.nan),
    ]
    == ["LOW", "MID", "HIGH", "UNKNOWN"],
)

# distance_pct 1% on close 7000 with ATR 14 -> 5.0 ATR
got = BM.pct_to_exec_atr(pd.Series([1.0]), 7000.0, 14.0)[0]
check("pct_to_exec_atr formula", abs(got - 5.0) < 1e-9, got)

# ============================================================
# 2) SMC directional pair: bull AND short, all three structures
# ============================================================
d = pd.DataFrame(
    {
        "trade_direction": [1.0, -1.0, 0.0],
        "internal_high_atr": [3.0, 3.0, 3.0],
        "internal_low_atr": [1.0, 1.0, 1.0],
        "swing_high_atr": [5.0, 5.0, 5.0],
        "swing_low_atr": [2.0, 2.0, 2.0],
        "ob_above_atr": [4.0, 4.0, 4.0],
        "ob_below_atr": [1.5, 1.5, 1.5],
    }
)
for kind in ("internal", "swing"):
    BM.directional_pair(
        d,
        high_col=f"{kind}_high_atr",
        low_col=f"{kind}_low_atr",
        out_forward=f"fwd_{kind}",
        out_backward=f"bwd_{kind}",
    )
BM.directional_pair(
    d,
    high_col="ob_above_atr",
    low_col="ob_below_atr",
    out_forward="fwd_ob",
    out_backward="bwd_ob",
)
for kind, hi, lo in (
    ("internal", 3.0, 1.0),
    ("swing", 5.0, 2.0),
    ("ob", 4.0, 1.5),
):
    check(
        f"{kind}: bull -> forward=high",
        d.loc[0, f"fwd_{kind}"] == hi
        and d.loc[0, f"bwd_{kind}"] == lo,
        (d.loc[0, f"fwd_{kind}"], d.loc[0, f"bwd_{kind}"]),
    )
    check(
        f"{kind}: short -> forward=low",
        d.loc[1, f"fwd_{kind}"] == lo
        and d.loc[1, f"bwd_{kind}"] == hi,
        (d.loc[1, f"fwd_{kind}"], d.loc[1, f"bwd_{kind}"]),
    )
    check(
        f"{kind}: skip -> nan",
        bool(np.isnan(d.loc[2, f"fwd_{kind}"]))
        and bool(np.isnan(d.loc[2, f"bwd_{kind}"])),
    )

b = pd.DataFrame(
    {"trade_direction": [1.0, -1.0, 0.0], "bias": [1.0, 1.0, 1.0]}
)
BM.add_relative_bias(b, "bias", "bias_rel")
check(
    "relative bias: follow +1 / fade -1 / skip 0",
    list(b["bias_rel"]) == [1.0, -1.0, 0.0],
    list(b["bias_rel"]),
)

# ============================================================
# 3) Reward: reuse of the validated Phase-1 simulator
# ============================================================
from research.analyze_ob_candidate_v3_phase1 import (  # noqa: E402
    build_path_arrays,
    rotated,
    simulate,
    apply_policy,
)


def make_path(cid: str, bars: list[tuple]) -> pd.DataFrame:
    rows = []
    for i, (o, h, l, c) in enumerate(bars, start=1):
        rows.append(
            {
                "candidate_id": cid,
                "step": i,
                "open_atr": o,
                "high_atr": h,
                "low_atr": l,
                "close_atr": c,
                "gap_minutes": 0.0 if i == 1 else 5.0,
            }
        )
    return pd.DataFrame(rows)


def run_one(bars: list[tuple], *, horizon=12):
    cid = "C0"
    cand = pd.DataFrame(
        {
            "candidate_id": [cid],
            "source_ob_bias": [1.0],
        }
    )
    path = make_path(cid, bars)
    arrays = build_path_arrays(cand, path)
    rot = rotated(arrays, np.array([1.0]))
    base, code = simulate(
        rot,
        arrays["contig"],
        stop_atr=SP.STOP_ATR,
        target_r=2.0,
        horizon=horizon,
        require_contiguous=True,
    )
    return apply_policy(
        base, code, 2.0, SP.SAME_BAR_POLICY
    ), code


# target hit
bars = [(0.0, 0.5, -0.2, 0.4)] * 24
bars[2] = (0.0, 2.5, -0.2, 2.4)
res, code = run_one(bars)
check(
    "reward: target hit",
    code[0] == EXIT_TARGET and abs(res[0] - 2.0) < 1e-9,
    (code[0], res[0]),
)

# stop hit
bars = [(0.0, 0.3, -0.2, -0.1)] * 24
bars[1] = (0.0, 0.4, -1.2, -1.1)
res, code = run_one(bars)
check(
    "reward: stop hit",
    code[0] == EXIT_STOP and abs(res[0] + 1.0) < 1e-9,
    (code[0], res[0]),
)

# timeout
res, code = run_one([(0.0, 0.2, -0.2, 0.1)] * 24)
check(
    "reward: timeout", code[0] == EXIT_TIMEOUT, (code[0], res[0])
)

# same-bar both hit -> conservative = -1
bars = [(0.0, 0.2, -0.2, 0.0)] * 24
bars[0] = (0.0, 2.5, -2.5, 0.0)
res, code = run_one(bars)
check(
    "reward: both-hit conservative -> -1",
    code[0] == EXIT_BOTH and abs(res[0] + 1.0) < 1e-9,
    (code[0], res[0]),
)

# gap stop
bars = [(0.0, 0.2, -0.2, 0.0)] * 24
bars[0] = (-1.5, 0.2, -1.6, -1.4)
res, code = run_one(bars)
check(
    "reward: gap stop",
    code[0] == EXIT_GAP_STOP and abs(res[0] + 1.5) < 1e-9,
    (code[0], res[0]),
)

# gap target
bars = [(0.0, 0.2, -0.2, 0.0)] * 24
bars[0] = (2.6, 2.7, 2.5, 2.65)
res, code = run_one(bars)
check(
    "reward: gap target",
    code[0] == EXIT_GAP_TARGET and abs(res[0] - 2.6) < 1e-9,
    (code[0], res[0]),
)


def run_noncontig():
    cid = "C0"
    cand = pd.DataFrame(
        {"candidate_id": [cid], "source_ob_bias": [1.0]}
    )
    p = make_path(cid, [(0.0, 0.2, -0.2, 0.0)] * 24)
    p.loc[p["step"] == 3, "gap_minutes"] = 60.0
    arrays = build_path_arrays(cand, p)
    rot = rotated(arrays, np.array([1.0]))
    base, code = simulate(
        rot,
        arrays["contig"],
        stop_atr=SP.STOP_ATR,
        target_r=2.0,
        horizon=12,
        require_contiguous=True,
    )
    return base, code


base, code = run_noncontig()
check(
    "reward: non-contiguous excluded",
    code[0] == EXIT_NONCONTIG and bool(np.isnan(base[0])),
    (code[0], base[0]),
)

# ============================================================
# 4) Synthetic inputs for the end-to-end builder run
# ============================================================
SYMBOLS = ["AG", "CU"]
N_PER = 10
N_CAND = len(SYMBOLS) * N_PER
BARS_PER_DAY = 48
DAY_STR = [f"2024-01-{d + 1:02d}" for d in range(6)]


def synth_five(symbol: str) -> pd.DataFrame:
    rows = []
    px = 7000.0 if symbol == "AG" else 50000.0
    for d in DAY_STR:
        for k in range(BARS_PER_DAY):
            start = (
                pd.Timestamp(d)
                + pd.Timedelta(hours=9)
                + pd.Timedelta(minutes=5 * k)
            )
            px += float(RNG.normal(0, 3.0))
            rows.append(
                {
                    "bar_start_time": start,
                    "bar_end_time": start
                    + pd.Timedelta(minutes=5),
                    "open": px,
                    "high": px + 5,
                    "low": px - 5,
                    "close": px + 1,
                    "trade": float(RNG.integers(100, 900)),
                    "position": float(RNG.integers(1000, 1900)),
                    "trading_day": d,
                }
            )
    return pd.DataFrame(rows)


RAW = {s: synth_five(s) for s in SYMBOLS}


def synth_candidates() -> pd.DataFrame:
    rows = []
    for si, s in enumerate(SYMBOLS):
        for i in range(N_PER):
            cid = f"{s}_C{si}_{i:04d}"
            g5 = bool(i % 2 == 0)
            g15 = bool(i % 3 == 0)
            g1 = bool(i % 5 == 0)
            if not (g5 or g15 or g1):
                g5 = True
            cnt = int(g5) + int(g15) + int(g1)
            breach = bool(i % 4 == 0)
            beyond = bool(i % 7 == 0)
            rows.append(
                {
                    "candidate_id": cid,
                    "candidate_group_id": f"{s}_G{i:04d}",
                    "symbol": s,
                    "source_tf": ["5m", "15m", "1h"][i % 3],
                    "source_ob_internal": bool(i % 2 == 0),
                    "source_ob_structure": "internal"
                    if i % 2 == 0
                    else "swing",
                    "source_ob_bias": 1.0 if i % 2 == 0 else -1.0,
                    "source_ob_zone_low": 6900.0,
                    "source_ob_zone_high": 7100.0,
                    "source_ob_width_atr5": 1.5,
                    "touch_ordinal": int(i % 4) + 1,
                    "is_first_touch": bool(i % 4 == 0),
                    "touch_intrabar_far_edge_breach": breach,
                    "touch_close_beyond_far_edge": beyond,
                    "touch_reclaimed_by_close": bool(
                        breach and not beyond
                    ),
                    "group_candidate_count": cnt,
                    "group_has_5m": g5,
                    "group_has_15m": g15,
                    "group_has_1h": g1,
                    "5m_atr14": 14.0,
                    "quant_width": 0.02
                    if i % 9
                    else np.nan,
                    "quant_width_percentile_train": 0.85
                    if i % 9
                    else np.nan,
                    "quant_top30_train": True if i % 9 else None,
                    "quant_crossed": False if i % 9 else None,
                }
            )
    return pd.DataFrame(rows)


CAND = synth_candidates()

TF_LEN = {"5m": len(RAW["AG"]), "15m": 96, "1h": 24}


def synth_context() -> pd.DataFrame:
    rows = []
    for _, c in CAND.iterrows():
        for tf in ("5m", "15m", "1h", "4h"):
            r = {
                "candidate_id": c["candidate_id"],
                "symbol": c["symbol"],
                "context_tf": tf,
                "bar_index": int(
                    RNG.integers(0, TF_LEN.get(tf, 20))
                )
                if tf != "4h"
                else 0,
                "close": 7000.0,
                "swing_bias": float(RNG.choice([-1, 0, 1])),
                "internal_bias": float(RNG.choice([-1, 0, 1])),
                "last_swing_structure_type": "BOS",
                "last_swing_structure_bias": 1.0,
                "last_swing_structure_age": 3.0,
                "last_internal_structure_type": "CHoCH",
                "last_internal_structure_bias": -1.0,
                "last_internal_structure_age": 5.0,
                "current_internal_high_distance_pct": 0.30,
                "current_internal_low_distance_pct": 0.20,
                "current_swing_high_distance_pct": 0.60,
                "current_swing_low_distance_pct": 0.40,
                "current_internal_high_level": 7100.0,
                "current_internal_low_level": 6900.0,
                "current_swing_high_level": 7200.0,
                "current_swing_low_level": 6800.0,
                "dsa_direction": float(RNG.choice([-1, 0, 1])),
                "dsa_raw_dsa_vwap_dev_pct": float(
                    RNG.uniform(-2, 2)
                ),
            }
            rows.append(r)
    return pd.DataFrame(rows)


CTX = synth_context()


def synth_levels() -> pd.DataFrame:
    rows = []
    ot_pool = [
        "active_bull_internal_ob",
        "active_bear_internal_ob",
        "active_bull_swing_ob",
        "active_bear_swing_ob",
        "current_internal_high",
        "EQH",
    ]
    for _, c in CAND.iterrows():
        for tf in ("5m", "15m", "1h"):
            for j in range(3):
                rel = ["above", "below", "overlap"][j]
                ot = ot_pool[(j + len(c["candidate_id"])) % len(ot_pool)]
                rows.append(
                    {
                        "event_id": c["candidate_id"],
                        "symbol": c["symbol"],
                        "timeframe": tf,
                        "object_type": ot,
                        "structure_class": "internal"
                        if "internal" in ot
                        else "swing"
                        if "swing" in ot
                        else "equal"
                        if ot.startswith("EQ")
                        else "internal",
                        "bias": 1.0 if j % 2 == 0 else -1.0,
                        "zone_low": 6950.0,
                        "zone_high": 7050.0,
                        "relation": rel,
                        "distance_pct": 0.10 * (j + 1),
                    }
                )
    return pd.DataFrame(rows)


LEVELS = synth_levels()


def synth_path() -> pd.DataFrame:
    rows = []
    for _, c in CAND.iterrows():
        for step in range(1, 25):
            o = float(RNG.normal(0, 0.1))
            rows.append(
                {
                    "candidate_id": c["candidate_id"],
                    "step": step,
                    "open_atr": o,
                    "high_atr": o + abs(RNG.normal(0, 0.4)),
                    "low_atr": o - abs(RNG.normal(0, 0.4)),
                    "close_atr": float(RNG.normal(0, 0.3)),
                    "gap_minutes": 0.0 if step == 1 else 5.0,
                }
            )
    return pd.DataFrame(rows)


PATH = synth_path()


def synth_momentum(bars: pd.DataFrame) -> pd.DataFrame:
    n = len(bars)
    return pd.DataFrame(
        {
            "bar_index": np.arange(n),
            "momentum_direction": RNG.choice(
                SP.MOMENTUM_DIRECTION_VALUES, n
            ),
            "sqzmom_val": RNG.normal(0, 1, n),
            "sqzmom_delta": RNG.normal(0, 1, n),
        }
    )


TABLES = {
    "candidates": CAND,
    "context": CTX,
    "levels": LEVELS,
    "path": PATH,
}
BM.load_full_or_chunks = lambda t: TABLES[t].copy()
BM.load_raw_five = lambda s: RAW[s].copy()
BM.build_momentum_frame = synth_momentum
BM.OUT_ROOT = TMP

# ============================================================
# 5) End-to-end run
# ============================================================
state = None
action_df = None
try:
    state, info = BM.build_state(
        CAND,
        CTX,
        LEVELS,
        {
            (s, tf): synth_momentum(RAW[s])
            for s in SYMBOLS
            for tf in SP.VALIDATED_TFS
        },
    )
    BM.audit_state_cardinality(state, CAND)
    BM.audit_state_columns(state)
    action_df = BM.expand_actions(state)
    action_df = BM.add_action_relative(action_df)
    action_df = BM.attach_rewards(action_df, CAND, PATH)
    BM.assert_action_cardinality(len(state), action_df)
    check("end-to-end build", True)
except Exception as e:  # pragma: no cover
    check("end-to-end build", False, repr(e))
    import traceback

    traceback.print_exc()
    raise SystemExit(1)

# ============================================================
# 6) Cardinality
# ============================================================
check("state rows == N", len(state) == N_CAND, (len(state), N_CAND))
check(
    "state candidate_id unique",
    bool(state["candidate_id"].is_unique),
)
check(
    "action rows == 7N",
    len(action_df) == N_CAND * len(SP.ACTIONS),
    (len(action_df), N_CAND * len(SP.ACTIONS)),
)
check(
    "no duplicate candidate-action",
    not action_df.duplicated(["candidate_id", "action"]).any(),
)
try:
    BM.assert_action_cardinality(N_CAND, action_df.iloc[1:])
    check("cardinality gate fires", False, "no raise")
except RuntimeError as e:
    check("cardinality gate fires", "mismatch" in str(e), str(e)[:80])

# ============================================================
# 7) Action-relative semantics on real pipeline output
# ============================================================
f = action_df[action_df["action"] == "FOLLOW_2.0R"].set_index(
    "candidate_id"
)
d_ = action_df[action_df["action"] == "FADE_2.0R"].set_index(
    "candidate_id"
)
for kind in ("internal", "swing"):
    ok = bool(
        np.allclose(
            f[f"forward_{kind}_atr_5m"].to_numpy(float),
            d_[f"backward_{kind}_atr_5m"].to_numpy(float),
            equal_nan=True,
        )
    )
    check(f"{kind}: follow-forward == fade-backward", ok)
ok = bool(
    np.allclose(
        f["forward_active_ob_atr_5m"].to_numpy(float),
        d_["backward_active_ob_atr_5m"].to_numpy(float),
        equal_nan=True,
    )
)
check("active OB: follow-forward == fade-backward", ok)

for tf in SP.VALIDATED_TFS:
    ok = bool(
        np.allclose(
            f[f"dsa_alignment_{tf}"].to_numpy(float),
            -d_[f"dsa_alignment_{tf}"].to_numpy(float),
            equal_nan=True,
        )
    )
    check(f"dsa_alignment {tf}: follow == -fade", ok)
    ok = bool(
        np.allclose(
            f[f"dsa_vwap_dev_rel_{tf}"].to_numpy(float),
            -d_[f"dsa_vwap_dev_rel_{tf}"].to_numpy(float),
            equal_nan=True,
        )
    )
    check(f"dsa_vwap_dev_rel {tf}: follow == -fade", ok)
    ok = bool(
        np.allclose(
            f[f"momentum_value_rel_{tf}"].to_numpy(float),
            -d_[f"momentum_value_rel_{tf}"].to_numpy(float),
            equal_nan=True,
        )
    )
    check(f"momentum_value_rel {tf}: follow == -fade", ok)
    ok = bool(
        np.allclose(
            f[f"momentum_delta_rel_{tf}"].to_numpy(float),
            -d_[f"momentum_delta_rel_{tf}"].to_numpy(float),
            equal_nan=True,
        )
    )
    check(f"momentum_delta_rel {tf}: follow == -fade", ok)

# momentum canonical string preserved
vals = set()
for tf in SP.VALIDATED_TFS:
    vals |= set(
        state[f"momentum_direction_{tf}"].astype(str).unique()
    )
check(
    "momentum canonical string preserved",
    vals.issubset(set(SP.MOMENTUM_DIRECTION_VALUES)),
    vals,
)

# ============================================================
# 8) Structural RR
# ============================================================
for tf in SP.VALIDATED_TFS:
    fw = f[f"forward_swing_atr_{tf}"].to_numpy(float)
    fit = f[f"target_fit_swing_{tf}"].to_numpy(float)
    ok = np.allclose(
        fit, np.where(np.isfinite(fw), fw / 2.0, np.nan),
        equal_nan=True,
    )
    check(f"target_fit_swing {tf} == forward/2R", bool(ok))
    bw = f[f"backward_internal_atr_{tf}"].to_numpy(float)
    st = f[f"stop_structure_internal_{tf}"].to_numpy(float)
    check(
        f"stop_structure_internal {tf} == backward/1ATR",
        bool(np.allclose(st, bw, equal_nan=True)),
    )
    ok = all(
        np.isnan(v)
        for v in action_df.loc[
            action_df["action"] == "SKIP",
            f"target_fit_ob_{tf}",
        ].to_numpy(float)
    )
    check(f"SKIP has no target_fit {tf}", ok)

# ============================================================
# 8a) State must stay ABSOLUTE (no action-relative encoding)
# ============================================================
ACTION_RELATIVE_MARKERS = (
    "forward_",
    "backward_",
    "_rel",
    "aligned",
    "opposed",
    "target_fit",
    "stop_structure",
    "trade_direction",
    "trade_mode",
    "action",
    "stop_atr",
    "target_R",
)
rel_in_state = sorted(
    {
        c
        for c in state.columns
        if any(m in c for m in ACTION_RELATIVE_MARKERS)
    }
)
check(
    "state contains no action-relative field",
    not rel_in_state,
    rel_in_state,
)
for tf in SP.VALIDATED_TFS:
    for need in (
        f"internal_bias_{tf}",
        f"swing_bias_{tf}",
        f"internal_high_atr_{tf}",
        f"internal_low_atr_{tf}",
        f"swing_high_atr_{tf}",
        f"swing_low_atr_{tf}",
        f"ob_above_atr_{tf}",
        f"ob_below_atr_{tf}",
        f"dsa_direction_{tf}",
        f"dsa_vwap_dev_pct_{tf}",
        f"momentum_direction_{tf}",
        f"sqzmom_val_{tf}",
        f"sqzmom_delta_{tf}",
        f"last_internal_structure_type_{tf}",
        f"last_swing_structure_age_{tf}",
    ):
        check(f"state has {need}", need in state.columns)

# ============================================================
# 8b) Reward sanity on the synthetic action table
# ============================================================
non_skip = action_df[action_df["action"] != "SKIP"]
skip = action_df[action_df["action"] == "SKIP"]
check(
    "SKIP reward is exactly 0",
    bool(
        (skip["gross_R_h12"].to_numpy(float) == 0.0).all()
    )
    and bool(
        (
            skip["exit_code_h12"].to_numpy(int)
            == SP.SKIP_EXIT_CODE
        ).all()
    ),
)
finite_frac = float(
    np.isfinite(non_skip["gross_R_h12"].to_numpy(float)).mean()
)
check(
    "non-SKIP rewards are populated",
    finite_frac > 0.5,
    finite_frac,
)
codes = set(
    non_skip["exit_code_h12"].astype(int).unique().tolist()
)
check(
    "exit codes within canonical range",
    codes.issubset(
        {
            EXIT_NONCONTIG,
            EXIT_GAP_STOP,
            EXIT_GAP_TARGET,
            EXIT_BOTH,
            EXIT_STOP,
            EXIT_TARGET,
            EXIT_TIMEOUT,
        }
    ),
    codes,
)
check(
    "primary_reward_R == gross_R_h12",
    bool(
        np.allclose(
            action_df["primary_reward_R"].to_numpy(float),
            action_df["gross_R_h12"].to_numpy(float),
            equal_nan=True,
        )
    ),
)
check(
    "reward_version is GROSS",
    set(action_df["reward_version"].astype(str).unique())
    == {SP.REWARD_VERSION},
)
check(
    "decision_weight carried into action rows",
    "decision_weight" in action_df.columns
    and float(action_df["decision_weight"].sum()) > 0,
)

# ============================================================
# 9) Causality
# ============================================================
leaky = state.copy()
leaky["entry_next_5m_open"] = 1.0
try:
    BM.audit_state_columns(leaky)
    check("leakage: entry_next_5m_open blocked", False, "no raise")
except RuntimeError as e:
    check(
        "leakage: entry_next_5m_open blocked",
        "leakage" in str(e),
        str(e)[:100],
    )

leaky2 = state.copy()
leaky2["gross_R_h12"] = 0.0
try:
    BM.audit_state_columns(leaky2)
    check("leakage: reward blocked", False, "no raise")
except RuntimeError:
    check("leakage: reward blocked", True)

check(
    "no forbidden column in state",
    not (
        set(SP.FORBIDDEN_STATE_COLUMNS) & set(state.columns)
    ),
    sorted(set(SP.FORBIDDEN_STATE_COLUMNS) & set(state.columns)),
)

# ============================================================
# 10) 4h quarantine
# ============================================================
try:
    SP.assert_validated_tf("4h")
    check("4h rejected by assert_validated_tf", False, "no raise")
except RuntimeError:
    check("4h rejected by assert_validated_tf", True)

check(
    "4h input rows were seen",
    info["4h_context_input_rows_seen"] > 0,
    info["4h_context_input_rows_seen"],
)
try:
    BM.audit_no_quarantined(state, action_df)
    check("no 4h in output", True)
except RuntimeError as e:
    check("no 4h in output", False, str(e))

bad = state.copy()
bad["source_tf"] = bad["source_tf"].astype(object)
bad.loc[bad.index[0], "source_tf"] = "4h"
try:
    BM.audit_no_quarantined(bad)
    check("4h value in output blocked", False, "no raise")
except RuntimeError:
    check("4h value in output blocked", True)

# ============================================================
# 11) Levels vocabulary authority
# ============================================================
bad_lv = LEVELS.copy()
bad_lv.loc[bad_lv.index[0], "object_type"] = "active_new_ob"
try:
    SP.assert_level_vocabulary(bad_lv)
    check("unknown level vocabulary blocked", False, "no raise")
except RuntimeError as e:
    check(
        "unknown level vocabulary blocked",
        "vocabulary drift" in str(e),
        str(e)[:100],
    )

check(
    "active OB types are the frozen 4",
    set(SP.ACTIVE_OB_TYPES)
    == {
        "active_bull_internal_ob",
        "active_bear_internal_ob",
        "active_bull_swing_ob",
        "active_bear_swing_ob",
    },
)

# ============================================================
print("\n=========== OB RL V0 SYNTHETIC RESULT ===========")
fails = [r for r in RESULTS if not r[1]]
for name, _, det in fails:
    print(f"FAIL  {name} :: {det}")
print(
    f"checks={len(RESULTS)} pass={len(RESULTS) - len(fails)} "
    f"fail={len(fails)}"
)
print("=================================================")
raise SystemExit(1 if fails else 0)
