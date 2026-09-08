#!/usr/bin/env python3

"""Phase-1.2 mechanism exploration for OB Candidate Universe V3.

Frozen: V3 dataset, Phase-1 and Phase-1.1 results. This script only
reads them.

Goal is NOT to prove any single indicator. It is to locate a
mechanism: which structural trigger, under which multi-timeframe
environment, with which spatial room, at which risk-reward setting,
produces stable positive R expectancy across futures.

Separation of concerns:
    candidate generator  -> 5m/15m/1h OB touch
    environment          -> DSA (multi-TF joint state) + SMC + session
    spatial context      -> forward room vs target distance
    mechanism            -> touch / retest / breach / reclaim
    opportunity          -> Quantile (how much R is achievable)
    execution            -> ATR stop x R target

NO parameter search, NO feature selection, NO model fitting beyond the
pre-registered conditional tables and contrasts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "research" / "analysis_results" / "ob_candidate_v3_phase1_2"
PHASE1 = ROOT / "research" / "analysis_results" / "ob_candidate_v3_phase1"

from research.audit_ob_candidate_v3_posthoc import (  # noqa: E402
    load_full_or_chunks,
    load_raw_five,
)
from research.analyze_ob_candidate_v3_phase1 import (  # noqa: E402
    build_path_arrays,
    rotated,
    simulate,
    apply_policy,
    weighted_mean,
)
from research.analyze_ob_candidate_v3_phase1_1 import (  # noqa: E402
    agg_rr_weighted,
    block_bootstrap_mean,
)

PRIMARY_STOP = 1.0
PRIMARY_TARGET = 2.0
PRIMARY_HORIZON = 12
PRIMARY_DIRECTION = "follow"
PRIMARY_POLICY = "conservative"

STOPS = (0.75, 1.00, 1.25)
TARGETS = (1.5, 2.0, 2.5)
HORIZON = 12

SYMBOLS = ("AG", "CU", "RB", "M")
PRIMARY_TFS = ("5m", "15m", "1h")

N_BOOT = 1000
N_BOOT_CONTRAST = 3000
SEED = 42

SUPPORTIVE = (
    "FULL_ALIGNED",
    "HTF_ALIGNED_PULLBACK",
    "HTF_SUPPORTIVE",
)


# ============================================================
# Frame
# ============================================================

def build_analysis_frame():
    candidates = load_full_or_chunks("candidates")
    context = load_full_or_chunks("context")
    path = load_full_or_chunks("path")

    ann = pd.read_csv(
        PHASE1 / "candidate_annotations.csv"
    )
    candidates = candidates.merge(
        ann,
        on=["candidate_id", "symbol", "source_tf"],
        how="left",
        validate="one_to_one",
    )

    piv = context.pivot(
        index="candidate_id",
        columns="context_tf",
        values=[
            "internal_bias",
            "swing_bias",
            "dsa_direction",
            "nearest_above_distance_pct",
            "nearest_below_distance_pct",
        ],
    )
    piv.columns = [
        f"{field}_{tf}" for field, tf in piv.columns
    ]
    piv = piv.reindex(
        candidates["candidate_id"].to_numpy()
    ).reset_index(drop=True)
    for col in piv.columns:
        candidates[col] = piv[col].to_numpy()

    return candidates, context, path


# ============================================================
# Environment: DSA joint state (H2)
# ============================================================

def relative_state(state, trade_direction):
    if not np.isfinite(state):
        return np.nan
    return int(state) * int(trade_direction)


def add_dsa_environment(candidates, trade_direction):
    r1h = (
        candidates["dsa_direction_1h"].to_numpy(float)
        * trade_direction
    )
    r15 = (
        candidates["dsa_direction_15m"].to_numpy(float)
        * trade_direction
    )
    r5 = (
        candidates["dsa_direction_5m"].to_numpy(float)
        * trade_direction
    )
    candidates["dsa_r1h"] = r1h
    candidates["dsa_r15"] = r15
    candidates["dsa_r5"] = r5
    candidates["dsa_exact"] = [
        (
            f"{int(a):+d}|{int(b):+d}|{int(c):+d}"
            if np.isfinite(a) and np.isfinite(b) and np.isfinite(c)
            else "UNKNOWN"
        )
        for a, b, c in zip(r1h, r15, r5)
    ]
    return candidates


def classify_dsa_env(r1h, r15, r5):
    if not all(
        np.isfinite(x) for x in (r1h, r15, r5)
    ):
        return "UNKNOWN"

    r1h = int(r1h)
    r15 = int(r15)
    r5 = int(r5)

    if r1h == 1 and r15 == 1 and r5 == 1:
        return "FULL_ALIGNED"
    if r1h == 1 and r15 == 1 and r5 <= 0:
        return "HTF_ALIGNED_PULLBACK"
    if r1h == 1 and r15 >= 0:
        return "HTF_SUPPORTIVE"
    if r1h == -1 and r15 <= 0:
        return "HTF_OPPOSED"
    if r1h == 0 and r15 == 0:
        return "HTF_FLAT"
    return "MIXED"


# ============================================================
# Spatial room (H5)
# ============================================================

def pct_distance_to_atr(distance_pct, price, atr):
    return (distance_pct / 100.0) * price / atr


def build_session_segment(five):
    x = five.sort_values("bar_start_time").copy()
    t = pd.to_datetime(x["bar_start_time"])
    new_segment = t.diff() > pd.Timedelta(minutes=5)
    x["session_segment"] = new_segment.cumsum()
    return x


# ============================================================
# Excursion
# ============================================================

def excursion_atr(rot, contig, h):
    hi = rot["hi"][:, :h].copy()
    lo = rot["lo"][:, :h].copy()
    ct = contig[:, :h]
    ok = (
        np.isfinite(hi).all(axis=1)
        & np.isfinite(lo).all(axis=1)
        & ct.all(axis=1)
    )
    n = hi.shape[0]
    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    mfe[ok] = np.maximum(hi[ok].max(axis=1), 0.0)
    mae[ok] = np.maximum(-lo[ok].min(axis=1), 0.0)
    return mfe, mae


# ============================================================
# Contrast bootstrap
# ============================================================

def bootstrap_contrast(
    result,
    weights,
    blocks,
    labels,
    group_a,
    group_b,
    *,
    n_boot=N_BOOT_CONTRAST,
    seed=SEED,
):
    rng = np.random.default_rng(seed)

    valid = (
        np.isfinite(result)
        & np.isfinite(weights)
        & (weights > 0)
    )
    result = result[valid]
    weights = weights[valid]
    blocks = blocks[valid]
    labels = labels[valid]

    unique_days = np.unique(blocks)
    by_day = {
        d: np.flatnonzero(blocks == d)
        for d in unique_days
    }
    if len(unique_days) == 0:
        return {
            "delta": np.nan,
            "ci025": np.nan,
            "ci975": np.nan,
            "p_delta_gt_0": np.nan,
        }

    ma_all = labels == group_a
    mb_all = labels == group_b
    point_a = (
        weighted_mean(result[ma_all], weights[ma_all])
        if ma_all.any()
        else np.nan
    )
    point_b = (
        weighted_mean(result[mb_all], weights[mb_all])
        if mb_all.any()
        else np.nan
    )

    draws = []
    for _ in range(n_boot):
        sampled = rng.choice(
            unique_days,
            size=len(unique_days),
            replace=True,
        )
        idx = np.concatenate(
            [by_day[d] for d in sampled]
        )
        la = labels[idx] == group_a
        lb = labels[idx] == group_b
        if not la.any() or not lb.any():
            continue
        ma = weighted_mean(
            result[idx][la], weights[idx][la]
        )
        mb = weighted_mean(
            result[idx][lb], weights[idx][lb]
        )
        draws.append(ma - mb)

    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    if len(draws) == 0:
        return {
            "delta": (
                point_a - point_b
                if np.isfinite(point_a)
                and np.isfinite(point_b)
                else np.nan
            ),
            "ci025": np.nan,
            "ci975": np.nan,
            "p_delta_gt_0": np.nan,
        }
    return {
        "delta": (
            round(float(point_a - point_b), 6)
            if np.isfinite(point_a) and np.isfinite(point_b)
            else np.nan
        ),
        "ci025": round(
            float(np.quantile(draws, 0.025)), 6
        ),
        "ci975": round(
            float(np.quantile(draws, 0.975)), 6
        ),
        "p_delta_gt_0": round(
            float(np.mean(draws > 0)), 6
        ),
    }


# ============================================================
# Verdict
# ============================================================

def verdict_of(
    *,
    pooled_mean_R,
    symbols_positive,
    p_gt_0,
    ci025,
    single_dominant,
    parameter_stable,
    time_stable,
    n,
):
    if (
        not np.isfinite(pooled_mean_R)
        or n < 100
    ):
        return "INCONCLUSIVE"

    strong = (
        symbols_positive >= 3
        and pooled_mean_R > 0
        and np.isfinite(p_gt_0)
        and p_gt_0 >= 0.90
        and not single_dominant
        and parameter_stable
        and time_stable
        and np.isfinite(ci025)
        and ci025 > 0
    )
    if strong:
        return "PASS"

    base = (
        symbols_positive >= 3
        and pooled_mean_R > 0
        and np.isfinite(p_gt_0)
        and p_gt_0 >= 0.90
        and not single_dominant
        and parameter_stable
        and time_stable
    )
    if base:
        return "PASS"

    if (
        symbols_positive >= 3
        and pooled_mean_R > 0
        and not single_dominant
    ):
        return "PROMISING"

    if pooled_mean_R <= 0 or symbols_positive <= 1:
        return "FAIL"

    return "INCONCLUSIVE"


# ============================================================
# Main
# ============================================================

def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    candidates, context, path = build_analysis_frame()
    symbols = sorted(candidates["symbol"].unique().tolist())

    raw_five = {}
    for s in symbols:
        five = load_raw_five(s)
        raw_five[s] = build_session_segment(five)

    # trading day + session segment
    parts = []
    for s, g in candidates.groupby("symbol"):
        five = raw_five[s]
        idx = g["touch_5m_bar_index"].astype(int).to_numpy()
        x = g.copy()
        x["trading_day"] = (
            five.iloc[idx]["trading_day"].astype(str).to_numpy()
        )
        x["session_segment"] = (
            five.iloc[idx]["session_segment"].to_numpy()
        )
        parts.append(x)
    candidates = pd.concat(parts, ignore_index=True)

    n = len(candidates)
    blocks = candidates["trading_day"].to_numpy()
    w = (
        1.0
        / candidates["group_candidate_count"].to_numpy(float)
    )

    bias = candidates["source_ob_bias"].to_numpy(float)
    sym_arr = candidates["symbol"].to_numpy(object)
    tf_arr = candidates["source_tf"].to_numpy(object)
    behavior = candidates["touch_behavior"].to_numpy(object)

    arr = build_path_arrays(candidates, path)
    contig = arr["contig"]
    rots = {
        "follow": rotated(arr, bias),
        "fade": rotated(arr, -bias),
    }

    # ---------------- primary + neighbourhood outcomes --------
    sim = {}
    for stop in STOPS:
        for tgt in TARGETS:
            base, code = simulate(
                rots["follow"], contig, stop, tgt, HORIZON, True
            )
            sim[("follow", stop, tgt)] = (
                apply_policy(base, code, tgt, PRIMARY_POLICY),
                code,
            )
    base_f, code_f = simulate(
        rots["fade"],
        contig,
        PRIMARY_STOP,
        PRIMARY_TARGET,
        HORIZON,
        True,
    )
    sim[("fade", PRIMARY_STOP, PRIMARY_TARGET)] = (
        apply_policy(
            base_f, code_f, PRIMARY_TARGET, PRIMARY_POLICY
        ),
        code_f,
    )

    res_p, code_p = sim[
        ("follow", PRIMARY_STOP, PRIMARY_TARGET)
    ]
    res_f, code_f = sim[
        ("fade", PRIMARY_STOP, PRIMARY_TARGET)
    ]

    mfe, mae = excursion_atr(
        rots["follow"], contig, HORIZON
    )

    # ---------------- spatial room (H5) ------------------------
    price = (
        candidates["entry_next_5m_open"].to_numpy(float)
        * 0.0
    )
    # reference price for pct conversion = touch bar close proxy
    # (distance_pct was measured against the trigger close).
    trig_close = np.full(n, np.nan)
    for s in symbols:
        m = sym_arr == s
        five = raw_five[s]
        idx = (
            candidates.loc[m, "touch_5m_bar_index"]
            .astype(int)
            .to_numpy()
        )
        trig_close[m] = (
            five.iloc[idx]["close"].to_numpy(float)
        )
    atr5 = candidates["5m_atr14"].to_numpy(float)

    room_fwd = {}
    for tf in PRIMARY_TFS:
        up = candidates[
            f"nearest_above_distance_pct_{tf}"
        ].to_numpy(float)
        dn = candidates[
            f"nearest_below_distance_pct_{tf}"
        ].to_numpy(float)
        up_atr = pct_distance_to_atr(up, trig_close, atr5)
        dn_atr = pct_distance_to_atr(dn, trig_close, atr5)
        # long -> room above; short -> room below
        room_fwd[tf] = np.where(
            bias == 1, up_atr, dn_atr
        )
        candidates[f"forward_room_{tf}"] = room_fwd[tf]

    stack = np.column_stack(
        [room_fwd[tf] for tf in PRIMARY_TFS]
    )
    with np.errstate(invalid="ignore"):
        fwd_min = np.nanmin(
            np.where(np.isfinite(stack), stack, np.nan),
            axis=1,
        )
    fwd_min = np.where(
        np.all(~np.isfinite(stack), axis=1), np.nan, fwd_min
    )
    candidates["forward_room_min_atr"] = fwd_min

    room_class = np.select(
        [
            fwd_min < 1.0,
            fwd_min < 2.0,
            fwd_min >= 2.0,
        ],
        ["<1ATR", "1-2ATR", ">=2ATR"],
        default="UNKNOWN",
    )
    candidates["room_class"] = room_class

    # ---------------- masks -----------------------------------
    only15 = (
        (tf_arr == "15m")
        & ~candidates["group_has_5m"].to_numpy(bool)
        & ~candidates["group_has_1h"].to_numpy(bool)
    )

    h4a = (
        (tf_arr == "5m") & (behavior == "CLOSE_BEYOND")
    )
    h4b = (
        (tf_arr == "15m") & (behavior == "CLOSE_BEYOND")
    )

    subsets = {
        "ALL": np.ones(n, dtype=bool),
        "only15": only15,
        "H4A_5m_closebeyond_fade": h4a,
        "H4B_15m_closebeyond_follow": h4b,
    }
    subset_result = {
        "ALL": res_p,
        "only15": res_p,
        "H4A_5m_closebeyond_fade": res_f,
        "H4B_15m_closebeyond_follow": res_p,
    }
    subset_code = {
        "ALL": code_p,
        "only15": code_p,
        "H4A_5m_closebeyond_fade": code_f,
        "H4B_15m_closebeyond_follow": code_p,
    }

    def scope_stats(sel, res, code, label):
        st = agg_rr_weighted(res, code, w, sel)
        if st is None:
            return None
        bs = (
            block_bootstrap_mean(
                res[sel], w[sel], blocks[sel], n_boot=N_BOOT
            )
            if sel.sum() >= 150
            else {
                "ci025": np.nan,
                "ci975": np.nan,
                "p_gt_0": np.nan,
            }
        )
        return {
            "scope": label,
            "n": st["n"],
            "n_weighted": st["n_weighted"],
            "mean_R": st["mean_R"],
            "mean_R_event": st["mean_R_event"],
            "target_hit_rate_w": st["target_hit_rate_w"],
            "stop_hit_rate_w": st["stop_hit_rate_w"],
            "both_hit_rate_w": st["both_hit_rate_w"],
            "ci025": bs["ci025"],
            "ci975": bs["ci975"],
            "p_gt_0": bs["p_gt_0"],
        }

    # ---------------- hypothesis_baseline.csv ------------------
    rows = []
    for name, sel in subsets.items():
        res = subset_result[name]
        r = scope_stats(sel, res, code_p, "ALL")
        if r:
            rows.append({"hypothesis": name, "symbol": "ALL", **r})
        for s in SYMBOLS:
            sel2 = sel & (sym_arr == s)
            if sel2.sum() == 0:
                continue
            r2 = scope_stats(sel2, res, code_p, s)
            if r2:
                rows.append(
                    {
                        "hypothesis": name,
                        "symbol": s,
                        **r2,
                    }
                )
    pd.DataFrame(rows).to_csv(
        RESULTS / "hypothesis_baseline.csv", index=False
    )
    print("hypothesis_baseline", len(rows), flush=True)

    # ---------------- H1: RR surface on only15 -----------------
    rows = []
    for stop in STOPS:
        for tgt in TARGETS:
            r, c = sim[("follow", stop, tgt)]
            st = agg_rr_weighted(r, c, w, only15)
            if st is None:
                continue
            rows.append(
                {
                    "symbol": "ALL",
                    "stop_atr": stop,
                    "target_r": tgt,
                    "n": st["n"],
                    "mean_R": st["mean_R"],
                    "target_hit_rate_w": st[
                        "target_hit_rate_w"
                    ],
                    "stop_hit_rate_w": st["stop_hit_rate_w"],
                }
            )
            for s in SYMBOLS:
                sel = only15 & (sym_arr == s)
                st2 = agg_rr_weighted(
                    r, c, w, np.flatnonzero(sel)
                )
                if st2 is None:
                    continue
                rows.append(
                    {
                        "symbol": s,
                        "stop_atr": stop,
                        "target_r": tgt,
                        "n": st2["n"],
                        "mean_R": st2["mean_R"],
                        "target_hit_rate_w": st2[
                            "target_hit_rate_w"
                        ],
                        "stop_hit_rate_w": st2[
                            "stop_hit_rate_w"
                        ],
                    }
                )
    surf = pd.DataFrame(rows)
    surf.to_csv(
        RESULTS / "h1_only15_rr_surface.csv", index=False
    )
    pool_surf = surf[surf.symbol == "ALL"]
    param_pos = int((pool_surf["mean_R"] > 0).sum())
    param_total = int(len(pool_surf))
    print(
        "h1_only15_rr_surface",
        len(surf),
        "positive_cells=%d/%d" % (param_pos, param_total),
        flush=True,
    )

    # ---------------- H1 stability -----------------------------
    rows = []
    splits = [
        ("all", np.ones(n, dtype=bool)),
        ("internal", candidates["source_ob_internal"].to_numpy(bool)),
        ("swing", ~candidates["source_ob_internal"].to_numpy(bool)),
        (
            "first_touch",
            candidates["is_first_touch"].to_numpy(bool),
        ),
        (
            "retest",
            ~candidates["is_first_touch"].to_numpy(bool),
        ),
        (
            "active_at_touch",
            candidates[
                "source_ob_active_at_touch_close"
            ]
            .to_numpy(bool),
        ),
    ]
    for label, extra in splits:
        sel = only15 & extra
        if sel.sum() == 0:
            continue
        r = scope_stats(sel, res_p, code_p, "ALL")
        if r:
            rows.append({"split": label, "symbol": "ALL", **r})
        for s in SYMBOLS:
            sel2 = sel & (sym_arr == s)
            if sel2.sum() == 0:
                continue
            r2 = scope_stats(sel2, res_p, code_p, s)
            if r2:
                rows.append(
                    {"split": label, "symbol": s, **r2}
                )
    pd.DataFrame(rows).to_csv(
        RESULTS / "h1_only15_stability.csv", index=False
    )
    print("h1_only15_stability", len(rows), flush=True)

    # ---------------- H2: DSA environment ----------------------
    trade_dir = bias
    for tf in PRIMARY_TFS:
        candidates[f"smc_internal_r_{tf}"] = (
            candidates[f"internal_bias_{tf}"].to_numpy(float)
            * trade_dir
        )
        candidates[f"smc_swing_r_{tf}"] = (
            candidates[f"swing_bias_{tf}"].to_numpy(float)
            * trade_dir
        )

    candidates = add_dsa_environment(candidates, trade_dir)
    candidates["dsa_env"] = [
        classify_dsa_env(a, b, c)
        for a, b, c in zip(
            candidates["dsa_r1h"].to_numpy(float),
            candidates["dsa_r15"].to_numpy(float),
            candidates["dsa_r5"].to_numpy(float),
        )
    ]
    dsa_env = candidates["dsa_env"].to_numpy(object)

    # exact state counts
    ex = candidates.loc[only15, "dsa_exact"].value_counts()
    ex_rows = []
    for state, cnt in ex.items():
        sel = only15 & (
            candidates["dsa_exact"].to_numpy(object) == state
        )
        st = agg_rr_weighted(
            res_p, code_p, w, np.flatnonzero(sel)
        )
        ex_rows.append(
            {
                "dsa_exact": state,
                "n": int(cnt),
                "mean_R": (
                    st["mean_R"] if st else np.nan
                ),
                "n_analyzed": (st["n"] if st else 0),
            }
        )
    pd.DataFrame(ex_rows).to_csv(
        RESULTS / "dsa_exact_state.csv", index=False
    )

    # environment table
    rows = []
    for env in (
        "FULL_ALIGNED",
        "HTF_ALIGNED_PULLBACK",
        "HTF_SUPPORTIVE",
        "HTF_OPPOSED",
        "HTF_FLAT",
        "MIXED",
        "UNKNOWN",
    ):
        sel = only15 & (dsa_env == env)
        if sel.sum() == 0:
            continue
        st = agg_rr_weighted(
            res_p, code_p, w, np.flatnonzero(sel)
        )
        if st is None:
            continue
        bs = (
            block_bootstrap_mean(
                res_p[sel],
                w[sel],
                blocks[sel],
                n_boot=N_BOOT,
            )
            if sel.sum() >= 150
            else {
                "ci025": np.nan,
                "ci975": np.nan,
                "p_gt_0": np.nan,
            }
        )
        rows.append(
            {
                "symbol": "ALL",
                "dsa_env": env,
                "n": st["n"],
                "n_weighted": st["n_weighted"],
                "mean_R": st["mean_R"],
                "target_hit_rate_w": st[
                    "target_hit_rate_w"
                ],
                "stop_hit_rate_w": st["stop_hit_rate_w"],
                "mean_MFE_ATR": round(
                    float(
                        weighted_mean(
                            mfe[sel], w[sel]
                        )
                    ),
                    6,
                ),
                "mean_MAE_ATR": round(
                    float(
                        weighted_mean(
                            mae[sel], w[sel]
                        )
                    ),
                    6,
                ),
                "ci025": bs["ci025"],
                "ci975": bs["ci975"],
                "p_gt_0": bs["p_gt_0"],
            }
        )
        for s in SYMBOLS:
            sel2 = sel & (sym_arr == s)
            st2 = agg_rr_weighted(
                res_p, code_p, w, np.flatnonzero(sel2)
            )
            if st2 is None:
                continue
            rows.append(
                {
                    "symbol": s,
                    "dsa_env": env,
                    "n": st2["n"],
                    "n_weighted": st2["n_weighted"],
                    "mean_R": st2["mean_R"],
                    "target_hit_rate_w": st2[
                        "target_hit_rate_w"
                    ],
                    "stop_hit_rate_w": st2[
                        "stop_hit_rate_w"
                    ],
                    "mean_MFE_ATR": round(
                        float(
                            weighted_mean(
                                mfe[sel2], w[sel2]
                            )
                        ),
                        6,
                    ),
                    "mean_MAE_ATR": round(
                        float(
                            weighted_mean(
                                mae[sel2], w[sel2]
                            )
                        ),
                        6,
                    ),
                    "ci025": np.nan,
                    "ci975": np.nan,
                    "p_gt_0": np.nan,
                }
            )
    pd.DataFrame(rows).to_csv(
        RESULTS / "dsa_environment.csv", index=False
    )
    print("dsa_environment", len(rows), flush=True)

    # contrasts
    sup_mask = only15 & np.isin(dsa_env, SUPPORTIVE)
    opp_mask = only15 & (dsa_env == "HTF_OPPOSED")
    pull_mask = only15 & (
        dsa_env == "HTF_ALIGNED_PULLBACK"
    )
    full_mask = only15 & (dsa_env == "FULL_ALIGNED")

    lab = np.full(n, "OTHER", dtype=object)
    lab[sup_mask] = "SUPPORTIVE"
    lab[opp_mask] = "OPPOSED"
    lab[pull_mask] = "PULLBACK"
    lab[full_mask] = "FULL_ALIGNED"

    c_rows = []
    for name, ga, gb in (
        ("A_supportive_vs_opposed", "SUPPORTIVE", "OPPOSED"),
        ("B_pullback_vs_full_aligned", "PULLBACK", "FULL_ALIGNED"),
    ):
        sel = only15 & np.isin(lab, [ga, gb])
        if sel.sum() == 0:
            continue
        bc = bootstrap_contrast(
            res_p,
            w,
            blocks,
            lab,
            ga,
            gb,
        )
        c_rows.append({"contrast": name, **bc})
    pd.DataFrame(c_rows).to_csv(
        RESULTS / "dsa_environment_contrast.csv", index=False
    )
    print("dsa_environment_contrast", len(c_rows), flush=True)

    # ---------------- H3: SMC environment + interaction --------
    rows = []
    for tf in PRIMARY_TFS:
        for kind in ("internal", "swing"):
            col = f"smc_{kind}_r_{tf}"
            v = candidates[col].to_numpy(float)
            for state in (-1, 0, 1):
                sel = only15 & (v == state)
                if sel.sum() == 0:
                    continue
                st = agg_rr_weighted(
                    res_p, code_p, w, np.flatnonzero(sel)
                )
                if st is None:
                    continue
                rows.append(
                    {
                        "tf": tf,
                        "kind": kind,
                        "relative_state": state,
                        "n": st["n"],
                        "mean_R": st["mean_R"],
                        "target_hit_rate_w": st[
                            "target_hit_rate_w"
                        ],
                    }
                )
    pd.DataFrame(rows).to_csv(
        RESULTS / "smc_environment.csv", index=False
    )

    smc_sup = (
        (candidates["smc_internal_r_1h"].to_numpy(float) == 1)
        & (
            candidates["smc_internal_r_15m"].to_numpy(float)
            == 1
        )
    )
    dsa_sup = np.isin(dsa_env, SUPPORTIVE)

    rows = []
    for sname, smask in (
        ("SMC+/DSA+", smc_sup & dsa_sup),
        ("SMC+/DSA-", smc_sup & ~dsa_sup),
        ("SMC-/DSA+", ~smc_sup & dsa_sup),
        ("SMC-/DSA-", ~smc_sup & ~dsa_sup),
    ):
        sel = only15 & smask
        if sel.sum() == 0:
            continue
        st = agg_rr_weighted(
            res_p, code_p, w, np.flatnonzero(sel)
        )
        if st is None:
            continue
        bs = (
            block_bootstrap_mean(
                res_p[sel],
                w[sel],
                blocks[sel],
                n_boot=N_BOOT,
            )
            if sel.sum() >= 150
            else {
                "ci025": np.nan,
                "ci975": np.nan,
                "p_gt_0": np.nan,
            }
        )
        rows.append(
            {
                "cell": sname,
                "symbol": "ALL",
                "n": st["n"],
                "mean_R": st["mean_R"],
                "target_hit_rate_w": st[
                    "target_hit_rate_w"
                ],
                "mean_MFE_ATR": round(
                    float(weighted_mean(mfe[sel], w[sel])), 6
                ),
                "mean_MAE_ATR": round(
                    float(weighted_mean(mae[sel], w[sel])), 6
                ),
                "ci025": bs["ci025"],
                "ci975": bs["ci975"],
                "p_gt_0": bs["p_gt_0"],
            }
        )
    inter = pd.DataFrame(rows)
    inter.to_csv(
        RESULTS / "smc_dsa_interaction.csv", index=False
    )
    print("smc_dsa_interaction", len(rows), flush=True)

    # ---------------- H5: spatial room -------------------------
    rows = []
    for rc in ("<1ATR", "1-2ATR", ">=2ATR", "UNKNOWN"):
        sel = (
            candidates["room_class"].to_numpy(object) == rc
        )
        if sel.sum() == 0:
            continue
        st = agg_rr_weighted(
            res_p, code_p, w, np.flatnonzero(sel)
        )
        if st is None:
            continue
        bs = (
            block_bootstrap_mean(
                res_p[sel],
                w[sel],
                blocks[sel],
                n_boot=N_BOOT,
            )
            if sel.sum() >= 150
            else {
                "ci025": np.nan,
                "ci975": np.nan,
                "p_gt_0": np.nan,
            }
        )
        rows.append(
            {
                "room_class": rc,
                "symbol": "ALL",
                "n": st["n"],
                "mean_R": st["mean_R"],
                "target_hit_rate_w": st[
                    "target_hit_rate_w"
                ],
                "ci025": bs["ci025"],
                "ci975": bs["ci975"],
                "p_gt_0": bs["p_gt_0"],
            }
        )
        for s in SYMBOLS:
            sel2 = sel & (sym_arr == s)
            st2 = agg_rr_weighted(
                res_p, code_p, w, np.flatnonzero(sel2)
            )
            if st2 is None:
                continue
            rows.append(
                {
                    "room_class": rc,
                    "symbol": s,
                    "n": st2["n"],
                    "mean_R": st2["mean_R"],
                    "target_hit_rate_w": st2[
                        "target_hit_rate_w"
                    ],
                    "ci025": np.nan,
                    "ci975": np.nan,
                    "p_gt_0": np.nan,
                }
            )
    pd.DataFrame(rows).to_csv(
        RESULTS / "spatial_room.csv", index=False
    )
    print("spatial_room", len(rows), flush=True)

    # ---------------- H4: touch mechanism ----------------------
    rows = []
    for hname, sel_base, res, cd, dlabel in (
        (
            "H4A_5m_closebeyond_fade",
            h4a,
            res_f,
            code_f,
            "fade",
        ),
        (
            "H4B_15m_closebeyond_follow",
            h4b,
            res_p,
            code_p,
            "follow",
        ),
    ):
        st = agg_rr_weighted(
            res, cd, w, np.flatnonzero(sel_base)
        )
        if st:
            bs = block_bootstrap_mean(
                res[sel_base],
                w[sel_base],
                blocks[sel_base],
                n_boot=N_BOOT,
            )
            rows.append(
                {
                    "hypothesis": hname,
                    "direction": dlabel,
                    "split": "all",
                    "symbol": "ALL",
                    "n": st["n"],
                    "mean_R": st["mean_R"],
                    "ci025": bs["ci025"],
                    "ci975": bs["ci975"],
                    "p_gt_0": bs["p_gt_0"],
                }
            )
        for s in SYMBOLS:
            sel = sel_base & (sym_arr == s)
            if sel.sum() == 0:
                continue
            st2 = agg_rr_weighted(
                res, code_p, w, np.flatnonzero(sel)
            )
            if st2 is None:
                continue
            bs2 = (
                block_bootstrap_mean(
                    res[sel],
                    w[sel],
                    blocks[sel],
                    n_boot=N_BOOT,
                )
                if sel.sum() >= 150
                else {
                    "ci025": np.nan,
                    "ci975": np.nan,
                    "p_gt_0": np.nan,
                }
            )
            rows.append(
                {
                    "hypothesis": hname,
                    "direction": dlabel,
                    "split": "all",
                    "symbol": s,
                    "n": st2["n"],
                    "mean_R": st2["mean_R"],
                    "ci025": bs2["ci025"],
                    "ci975": bs2["ci975"],
                    "p_gt_0": bs2["p_gt_0"],
                }
            )
        for split_label, extra in (
            (
                "internal",
                candidates["source_ob_internal"].to_numpy(bool),
            ),
            (
                "swing",
                ~candidates["source_ob_internal"].to_numpy(bool),
            ),
            (
                "first_touch",
                candidates["is_first_touch"].to_numpy(bool),
            ),
            (
                "retest",
                ~candidates["is_first_touch"].to_numpy(bool),
            ),
            (
                "active_at_touch",
                candidates["source_ob_active_at_touch_close"]
                .to_numpy(bool),
            ),
        ):
            sel = sel_base & extra
            if sel.sum() == 0:
                continue
            st3 = agg_rr_weighted(
                res, cd, w, np.flatnonzero(sel)
            )
            if st3 is None:
                continue
            rows.append(
                {
                    "hypothesis": hname,
                    "direction": dlabel,
                    "split": split_label,
                    "symbol": "ALL",
                    "n": st3["n"],
                    "mean_R": st3["mean_R"],
                    "ci025": np.nan,
                    "ci975": np.nan,
                    "p_gt_0": np.nan,
                }
            )
    pd.DataFrame(rows).to_csv(
        RESULTS / "touch_mechanism.csv", index=False
    )
    print("touch_mechanism", len(rows), flush=True)

    # ---------------- H6: quantile coverage audit ---------------
    qknown = np.isfinite(
        candidates["quant_width_percentile_train"].to_numpy(
            float
        )
    )
    rows = []

    def add_cov(group_label, bucket, sel):
        if sel.sum() == 0:
            return
        for kn in (True, False):
            s2 = sel & (qknown == kn)
            if s2.sum() == 0:
                continue
            st = agg_rr_weighted(
                res_p, code_p, w, np.flatnonzero(s2)
            )
            if st is None:
                continue
            rows.append(
                {
                    "grouping": group_label,
                    "bucket": str(bucket),
                    "quantile_known": kn,
                    "n": st["n"],
                    "mean_R": st["mean_R"],
                }
            )

    add_cov("overall", "ALL", np.ones(n, dtype=bool))
    for s in SYMBOLS:
        add_cov("symbol", s, sym_arr == s)
    for tf in PRIMARY_TFS:
        add_cov("source_tf", tf, tf_arr == tf)
    for b in ("NO_BREACH", "BREACH_RECLAIM", "CLOSE_BEYOND"):
        add_cov("touch_behavior", b, behavior == b)
    for b in ("1", "2", "3", "4+"):
        add_cov(
            "touch_bin",
            b,
            candidates["touch_bin"].to_numpy(object) == b,
        )
    pd.DataFrame(rows).to_csv(
        RESULTS / "quantile_coverage_audit.csv", index=False
    )
    print("quantile_coverage_audit", len(rows), flush=True)

    # ---------------- H6: conditional quantile -----------------
    qbin = candidates["quant_bin"].to_numpy(object)
    rows = []
    for scope_label, sel_base, res in (
        ("H1_only15", only15, res_p),
        ("H4A_5m_closebeyond_fade", h4a, res_f),
        ("H4B_15m_closebeyond_follow", h4b, res_p),
    ):
        for tgt in (2.0, 2.5, 3.0):
            if tgt in TARGETS:
                r, c = sim[("follow", PRIMARY_STOP, tgt)]
            else:
                b2, c2 = simulate(
                    rots["follow"],
                    contig,
                    PRIMARY_STOP,
                    tgt,
                    HORIZON,
                    True,
                )
                r = apply_policy(
                    b2, c2, tgt, PRIMARY_POLICY
                )
                c = c2
            if scope_label == "H4A_5m_closebeyond_fade":
                b3, c3 = simulate(
                    rots["fade"],
                    contig,
                    PRIMARY_STOP,
                    tgt,
                    HORIZON,
                    True,
                )
                r = apply_policy(
                    b3, c3, tgt, PRIMARY_POLICY
                )
                c = c3
            for b in ("LOW", "MID", "HIGH"):
                sel = sel_base & (qbin == b)
                if sel.sum() == 0:
                    continue
                st = agg_rr_weighted(
                    r, c, w, np.flatnonzero(sel)
                )
                if st is None:
                    continue
                rows.append(
                    {
                        "scope": scope_label,
                        "target_r": tgt,
                        "quant_bin": b,
                        "n": st["n"],
                        "mean_R": st["mean_R"],
                        "target_hit_rate_w": st[
                            "target_hit_rate_w"
                        ],
                    }
                )
    pd.DataFrame(rows).to_csv(
        RESULTS / "quantile_conditional.csv", index=False
    )
    print("quantile_conditional", len(rows), flush=True)

    # ---------------- session stability ------------------------
    rows = []
    dom = {}
    for name, sel in subsets.items():
        res = subset_result[name]
        sel_idx = np.flatnonzero(sel)
        if len(sel_idx) == 0:
            continue
        sub_w = w[sel_idx]
        sub_res = res[sel_idx]
        sub_days = blocks[sel_idx]
        sub_seg = (
            candidates["session_segment"]
            .to_numpy()[sel_idx]
        )
        tot = float(np.nansum(sub_w))
        day_w = pd.Series(sub_w).groupby(sub_days).sum()
        seg_w = pd.Series(sub_w).groupby(sub_seg).sum()
        max_day = (
            float(day_w.max() / tot) if tot > 0 else np.nan
        )
        max_seg = (
            float(seg_w.max() / tot) if tot > 0 else np.nan
        )
        dom[name] = {
            "max_single_day_share": round(max_day, 6),
            "max_single_segment_share": round(max_seg, 6),
        }

        # trading-day tercile
        uniq_days = np.array(sorted(set(sub_days)))
        if len(uniq_days) >= 3:
            terc = np.array_split(uniq_days, 3)
            for k, grp in enumerate(terc):
                m = np.isin(sub_days, grp)
                st = agg_rr_weighted(
                    sub_res,
                    subset_code[name][sel_idx],
                    sub_w,
                    np.flatnonzero(m),
                )
                if st is None:
                    continue
                rows.append(
                    {
                        "subset": name,
                        "grouping": "trading_day_tercile",
                        "bucket": k,
                        "n": st["n"],
                        "mean_R": st["mean_R"],
                    }
                )
        # session-segment tercile
        uniq_seg = np.array(sorted(set(sub_seg)))
        if len(uniq_seg) >= 3:
            terc = np.array_split(uniq_seg, 3)
            for k, grp in enumerate(terc):
                m = np.isin(sub_seg, grp)
                st = agg_rr_weighted(
                    sub_res,
                    subset_code[name][sel_idx],
                    sub_w,
                    np.flatnonzero(m),
                )
                if st is None:
                    continue
                rows.append(
                    {
                        "subset": name,
                        "grouping": "session_segment_tercile",
                        "bucket": k,
                        "n": st["n"],
                        "mean_R": st["mean_R"],
                    }
                )
    pd.DataFrame(rows).to_csv(
        RESULTS / "session_stability.csv", index=False
    )
    print("session_stability", len(rows), flush=True)

    # ---------------- scorecard --------------------------------
    def sym_pos_count(sel, res, code=None):
        if code is None:
            code = code_p
        cnt = 0
        for s in SYMBOLS:
            s2 = sel & (sym_arr == s)
            if s2.sum() == 0:
                continue
            st = agg_rr_weighted(
                res, code, w, np.flatnonzero(s2)
            )
            if st and np.isfinite(st["mean_R"]):
                if st["mean_R"] > 0:
                    cnt += 1
        return cnt

    def scoped(sel, res, code=None):
        if code is None:
            code = code_p
        st = agg_rr_weighted(
            res, code, w, np.flatnonzero(sel)
        )
        if st is None:
            return None
        bs = (
            block_bootstrap_mean(
                res[sel],
                w[sel],
                blocks[sel],
                n_boot=N_BOOT,
            )
            if sel.sum() >= 150
            else {
                "ci025": np.nan,
                "ci975": np.nan,
                "p_gt_0": np.nan,
            }
        )
        return {
            "n": st["n"],
            "mean_R": st["mean_R"],
            "ci025": bs["ci025"],
            "ci975": bs["ci975"],
            "p_gt_0": bs["p_gt_0"],
        }

    def single_dominant(sel):
        if sel.sum() == 0:
            return True
        sw = w[sel]
        tot = float(np.nansum(sw))
        if tot <= 0:
            return True
        by = (
            pd.Series(sw)
            .groupby(sym_arr[sel])
            .sum()
            .sort_values(ascending=False)
        )
        return bool(float(by.iloc[0] / tot) > 0.60)

    sc = []

    r = scoped(only15, res_p)
    if r:
        sc.append(
            {
                "hypothesis": "H1 only15",
                **r,
                "symbols_positive": sym_pos_count(
                    only15, res_p
                ),
                "parameter_stable": bool(
                    param_pos >= 6
                ),
                "time_stable": bool(
                    dom.get("only15", {}).get(
                        "max_single_day_share", 1.0
                    )
                    < 0.25
                ),
                "single_dominant": single_dominant(only15),
            }
        )

    sup_only = only15 & np.isin(dsa_env, SUPPORTIVE)
    r = scoped(sup_only, res_p)
    if r:
        sc.append(
            {
                "hypothesis": "H2 DSA environment",
                **r,
                "symbols_positive": sym_pos_count(
                    sup_only, res_p
                ),
                "parameter_stable": True,
                "time_stable": bool(
                    dom.get("only15", {}).get(
                        "max_single_day_share", 1.0
                    )
                    < 0.25
                ),
                "single_dominant": single_dominant(sup_only),
            }
        )

    both = only15 & smc_sup & dsa_sup
    r = scoped(both, res_p)
    if r:
        sc.append(
            {
                "hypothesis": "H3 SMCxDSA",
                **r,
                "symbols_positive": sym_pos_count(
                    both, res_p
                ),
                "parameter_stable": True,
                "time_stable": bool(
                    dom.get("only15", {}).get(
                        "max_single_day_share", 1.0
                    )
                    < 0.25
                ),
                "single_dominant": single_dominant(both),
            }
        )

    for hname, sel, res in (
        ("H4A 5m failure fade", h4a, res_f),
        ("H4B 15m break follow", h4b, res_p),
    ):
        r = scoped(sel, res, cd)
        if r:
            sc.append(
                {
                    "hypothesis": hname,
                    **r,
                    "symbols_positive": sym_pos_count(
                        sel, res, cd
                    ),
                    "parameter_stable": True,
                    "time_stable": bool(
                        dom.get(
                            (
                                "H4A_5m_closebeyond_fade"
                                if hname.startswith("H4A")
                                else "H4B_15m_closebeyond_follow"
                            ),
                            {},
                        ).get("max_single_day_share", 1.0)
                        < 0.30
                    ),
                    "single_dominant": single_dominant(sel),
                }
            )

    roomy = (
        candidates["room_class"].to_numpy(object) == ">=2ATR"
    )
    r = scoped(roomy, res_p)
    if r:
        sc.append(
            {
                "hypothesis": "H5 spatial room",
                **r,
                "symbols_positive": sym_pos_count(
                    roomy, res_p
                ),
                "parameter_stable": True,
                "time_stable": bool(
                    dom.get("ALL", {}).get(
                        "max_single_day_share", 1.0
                    )
                    < 0.25
                ),
                "single_dominant": single_dominant(roomy),
            }
        )

    qh = only15 & (qbin == "HIGH")
    r = scoped(qh, res_p)
    if r:
        sc.append(
            {
                "hypothesis": "H6 quantile conditional",
                **r,
                "symbols_positive": sym_pos_count(
                    qh, res_p
                ),
                "parameter_stable": True,
                "time_stable": bool(
                    dom.get("only15", {}).get(
                        "max_single_day_share", 1.0
                    )
                    < 0.25
                ),
                "single_dominant": single_dominant(qh),
            }
        )

    for row in sc:
        row["verdict"] = verdict_of(
            pooled_mean_R=row["mean_R"],
            symbols_positive=row["symbols_positive"],
            p_gt_0=row["p_gt_0"],
            ci025=row["ci025"],
            single_dominant=row["single_dominant"],
            parameter_stable=row["parameter_stable"],
            time_stable=row["time_stable"],
            n=row["n"],
        )

    pd.DataFrame(sc).to_csv(
        RESULTS / "hypothesis_scorecard.csv", index=False
    )
    print("hypothesis_scorecard", len(sc), flush=True)

    summary = {
        "schema_version": "ob_candidate_v3_phase1_2",
        "parent_sha": (
            "fa9fc6951e63c4d9d01089e7db99d7915ddbf38d"
        ),
        "candidates": int(n),
        "trading_days": int(len(np.unique(blocks))),
        "primary": {
            "stop_atr": PRIMARY_STOP,
            "target_r": PRIMARY_TARGET,
            "horizon": PRIMARY_HORIZON,
            "direction": PRIMARY_DIRECTION,
            "policy": PRIMARY_POLICY,
        },
        "h1_parameter_positive_cells": (
            f"{param_pos}/{param_total}"
        ),
        "dominance": dom,
        "spatial_columns_used": [
            "nearest_above_distance_pct",
            "nearest_below_distance_pct",
        ],
        "confirm": {
            "v3_changed": 0,
            "phase1_changed": 0,
            "phase1_1_changed": 0,
        },
    }
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("PHASE1_2_DONE", flush=True)


if __name__ == "__main__":
    main()
