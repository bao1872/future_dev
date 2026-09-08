#!/usr/bin/env python3

"""Phase-1.1 robustness audit for OB Candidate Universe V3.

Frozen inputs: V3 dataset and the Phase-1 results. This script only
reads them and writes to ``analysis_results/ob_candidate_v3_phase1_1``.

Fixes four statistical-definition issues found while reviewing Phase-1:

  A  all rates are decision-bar weighted (event-weighted kept as
     robustness); mean_R was already weighted, rates were not.
  B  ordering ambiguity is judged PER CONFIGURATION, never averaged
     over the whole grid (0.5 ATR stops were being masked by 2 ATR
     stops).
  C  Quantile HIGH-vs-LOW is measured inside symbol x OOS fold, so a
     pooled monotonicity cannot be produced by symbol mix
     (Simpson's paradox).
  D/E cluster bootstrap resamples TRADING DAYS, not candidates
     (candidates sharing a 5m decision bar are not independent).
  H  DSA direction gets one final, methodologically clean test:
     group-locked folds, 60-minute embargo, sample-weighted fit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "research" / "analysis_results" / "ob_candidate_v3_phase1_1"
PHASE1 = ROOT / "research" / "analysis_results" / "ob_candidate_v3_phase1"

from research.audit_ob_candidate_v3_posthoc import (  # noqa: E402
    load_full_or_chunks,
    add_true_continuity,
    load_raw_five,
)
from research.analyze_ob_candidate_v3_phase1 import (  # noqa: E402
    build_path_arrays,
    rotated,
    simulate,
    apply_policy,
    weighted_mean,
    weighted_quantile,
    EXIT_INSUFFICIENT,
    EXIT_NONCONTIG,
    EXIT_GAP_STOP,
    EXIT_GAP_TARGET,
    EXIT_BOTH,
    EXIT_STOP,
    EXIT_TARGET,
    EXIT_TIMEOUT,
)

SYMBOLS = ("AG", "CU", "RB", "M")
PRIMARY_CONTEXT_TFS = ("5m", "15m", "1h")
DIRECTIONS = ("follow", "fade")
STOPS_ATR = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
TARGET_R = (1.0, 1.5, 2.0, 2.5, 3.0)
HORIZONS = (6, 12, 24)
PRIMARY_POLICY = "conservative"

REF_STOP = 1.00
REF_TARGET = 2.0
REF_HORIZON = 12

QUANT_STOPS = (0.75, 1.00, 1.25)
QUANT_TARGETS = (1.5, 2.0, 2.5, 3.0)
QBINS = ("LOW", "MID", "HIGH")

N_BOOT = 2000
SEED = 42


# ============================================================
# A - weighted rates
# ============================================================

def weighted_rate(
    mask: np.ndarray,
    analyzed: np.ndarray,
    w: np.ndarray,
) -> float:
    ok = analyzed & np.isfinite(w) & (w > 0)
    if not ok.any():
        return float("nan")
    return float(
        np.sum(w[ok] * mask[ok].astype(float))
        / np.sum(w[ok])
    )


def agg_rr_weighted(
    result: np.ndarray,
    code: np.ndarray,
    w: np.ndarray,
    idx: np.ndarray | None = None,
) -> dict | None:
    if idx is not None:
        result = result[idx]
        code = code[idx]
        w = w[idx]

    analyzed = (code != EXIT_INSUFFICIENT) & (
        code != EXIT_NONCONTIG
    )
    n_analyzed = int(analyzed.sum())
    if n_analyzed == 0:
        return None

    valid = analyzed & np.isfinite(result)
    n_valid = int(valid.sum())

    out = {
        "n": n_analyzed,
        "n_weighted": round(
            float(np.sum(w[analyzed])), 4
        ),
        "n_result": n_valid,
    }

    if n_valid == 0:
        out.update(
            {
                k: np.nan
                for k in (
                    "mean_R",
                    "mean_R_event",
                    "median_R",
                    "positive_R_rate_w",
                    "positive_R_rate_event",
                    "p10_R",
                    "p90_R",
                )
            }
        )
    else:
        r = result[valid]
        ww = w[valid]
        pos = r > 0
        out.update(
            {
                "mean_R": round(weighted_mean(r, ww), 6),
                "mean_R_event": round(float(np.mean(r)), 6),
                "median_R": round(float(np.median(r)), 6),
                "positive_R_rate_w": round(
                    weighted_rate(
                        pos,
                        np.ones_like(pos),
                        ww,
                    ),
                    6,
                ),
                "positive_R_rate_event": round(
                    float(np.mean(pos)), 6
                ),
                "p10_R": round(
                    weighted_quantile(r, 0.10, ww), 6
                ),
                "p90_R": round(
                    weighted_quantile(r, 0.90, ww), 6
                ),
            }
        )

    tgt = (code == EXIT_TARGET) | (code == EXIT_GAP_TARGET)
    stp = (code == EXIT_STOP) | (code == EXIT_GAP_STOP)
    tmo = code == EXIT_TIMEOUT
    both = code == EXIT_BOTH
    gstp = code == EXIT_GAP_STOP

    for name, mask in (
        ("target_hit_rate", tgt),
        ("stop_hit_rate", stp),
        ("timeout_rate", tmo),
        ("both_hit_rate", both),
        ("gap_stop_rate", gstp),
    ):
        out[name + "_w"] = round(
            weighted_rate(mask, analyzed, w), 6
        )
        out[name + "_event"] = round(
            float(mask[analyzed].mean()), 6
        )
    return out


# ============================================================
# B - per-configuration ambiguity gate
# ============================================================

def ambiguity_status(rate: float) -> str:
    if not np.isfinite(rate):
        return "UNKNOWN"
    if rate <= 0.02:
        return "5m_OK"
    if rate <= 0.10:
        return "BORDERLINE"
    return "NEEDS_1M"


# ============================================================
# E - trading-day attachment + cluster bootstrap
# ============================================================

def attach_trading_day(
    candidates: pd.DataFrame,
    raw_five_by_symbol: dict,
) -> pd.DataFrame:
    out = []
    for symbol, g in candidates.groupby("symbol"):
        five = raw_five_by_symbol[symbol]
        idx = (
            g["touch_5m_bar_index"].astype(int).to_numpy()
        )
        x = g.copy()
        x["trading_day"] = (
            five.iloc[idx]["trading_day"]
            .astype(str)
            .to_numpy()
        )
        out.append(x)
    return pd.concat(out, ignore_index=True)


def block_bootstrap_mean(
    values: np.ndarray,
    weights: np.ndarray,
    blocks: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    rng = np.random.default_rng(seed)
    unique_blocks = np.unique(blocks)
    block_to_idx = {
        b: np.flatnonzero(blocks == b)
        for b in unique_blocks
    }

    ok = np.isfinite(values) & np.isfinite(weights)
    values = values[ok]
    weights = weights[ok]
    blocks_ok = blocks[ok]
    block_to_idx = {
        b: np.flatnonzero(blocks_ok == b)
        for b in np.unique(blocks_ok)
    }
    ub = np.array(list(block_to_idx.keys()))
    if len(ub) == 0:
        return {
            "mean": np.nan,
            "ci025": np.nan,
            "ci975": np.nan,
            "p_gt_0": np.nan,
        }

    samples = []
    for _ in range(n_boot):
        picked = rng.choice(
            ub, size=len(ub), replace=True
        )
        idx = np.concatenate(
            [block_to_idx[b] for b in picked]
        )
        samples.append(
            weighted_mean(values[idx], weights[idx])
        )
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return {
            "mean": np.nan,
            "ci025": np.nan,
            "ci975": np.nan,
            "p_gt_0": np.nan,
        }
    return {
        "mean": round(
            weighted_mean(values, weights), 6
        ),
        "ci025": round(
            float(np.quantile(samples, 0.025)), 6
        ),
        "ci975": round(
            float(np.quantile(samples, 0.975)), 6
        ),
        "p_gt_0": round(float(np.mean(samples > 0)), 6),
    }


def block_bootstrap_grouped(
    values: np.ndarray,
    weights: np.ndarray,
    blocks: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    """Bootstrap several groups on the SAME resampled blocks so the
    between-group difference has a valid interval."""
    rng = np.random.default_rng(seed)
    ok = np.isfinite(values) & np.isfinite(weights)
    values = values[ok]
    weights = weights[ok]
    blocks_ok = blocks[ok]
    groups_ok = groups[ok]

    ub = np.unique(blocks_ok)
    if len(ub) == 0:
        return {}

    block_to_idx = {
        b: np.flatnonzero(blocks_ok == b) for b in ub
    }
    glabels = np.unique(groups_ok)

    samples = {g: [] for g in glabels}
    delta_samples = []

    for _ in range(n_boot):
        picked = rng.choice(
            ub, size=len(ub), replace=True
        )
        idx = np.concatenate(
            [block_to_idx[b] for b in picked]
        )
        v = values[idx]
        ww = weights[idx]
        gg = groups_ok[idx]
        means = {}
        for g in glabels:
            m = gg == g
            if m.any():
                means[g] = weighted_mean(v[m], ww[m])
            else:
                means[g] = np.nan
            samples[g].append(means[g])
        if (
            "HIGH" in means
            and "LOW" in means
            and np.isfinite(means["HIGH"])
            and np.isfinite(means["LOW"])
        ):
            delta_samples.append(
                means["HIGH"] - means["LOW"]
            )

    out = {}
    for g in glabels:
        s = np.asarray(samples[g], dtype=float)
        s = s[np.isfinite(s)]
        m = groups_ok == g
        point = (
            weighted_mean(values[m], weights[m])
            if m.any()
            else np.nan
        )
        out[g] = {
            "mean": (
                round(float(point), 6)
                if np.isfinite(point)
                else np.nan
            ),
            "ci025": (
                round(float(np.quantile(s, 0.025)), 6)
                if len(s)
                else np.nan
            ),
            "ci975": (
                round(float(np.quantile(s, 0.975)), 6)
                if len(s)
                else np.nan
            ),
        }
    d = np.asarray(delta_samples, dtype=float)
    d = d[np.isfinite(d)]
    if len(d):
        out["delta_HIGH_minus_LOW"] = {
            "mean": round(float(np.mean(d)), 6),
            "ci025": round(
                float(np.quantile(d, 0.025)), 6
            ),
            "ci975": round(
                float(np.quantile(d, 0.975)), 6
            ),
            "p_gt_0": round(float(np.mean(d > 0)), 6),
        }
    return out


# ============================================================
# H - purged, group-locked expanding splits
# ============================================================

def purged_expanding_splits(
    candidates: pd.DataFrame,
    *,
    n_folds: int = 5,
    embargo_minutes: int = 60,
):
    groups = (
        candidates[["candidate_group_id", "touch_time"]]
        .drop_duplicates("candidate_group_id")
        .sort_values("touch_time")
        .reset_index(drop=True)
    )
    chunks = np.array_split(
        np.arange(len(groups)), n_folds
    )
    touch = pd.to_datetime(groups["touch_time"])
    gid = candidates["candidate_group_id"].to_numpy()

    for fold in range(1, n_folds):
        test_groups = groups.iloc[chunks[fold]]
        if len(test_groups) == 0:
            continue
        test_start = pd.Timestamp(
            test_groups["touch_time"].min()
        )
        train_cutoff = test_start - pd.Timedelta(
            minutes=embargo_minutes
        )
        train_group_ids = set(
            groups.loc[
                touch < train_cutoff,
                "candidate_group_id",
            ]
        )
        test_group_ids = set(
            test_groups["candidate_group_id"]
        )
        if not train_group_ids:
            continue
        tr = np.flatnonzero(
            pd.Index(gid).isin(train_group_ids)
        )
        te = np.flatnonzero(
            pd.Index(gid).isin(test_group_ids)
        )
        if len(tr) == 0 or len(te) == 0:
            continue
        yield fold, tr, te


# ============================================================
# Main
# ============================================================

def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    candidates = load_full_or_chunks("candidates")
    path = load_full_or_chunks("path")
    context = load_full_or_chunks("context")
    ann = pd.read_csv(
        PHASE1 / "candidate_annotations.csv"
    )
    candidates = candidates.merge(
        ann,
        on=["candidate_id", "symbol", "source_tf"],
        how="left",
        validate="one_to_one",
    )

    symbols = sorted(candidates["symbol"].unique().tolist())
    raw_five = {s: load_raw_five(s) for s in symbols}
    candidates = attach_trading_day(candidates, raw_five)

    n = len(candidates)
    w = (
        1.0
        / candidates["group_candidate_count"].to_numpy(float)
    )
    blocks = candidates["trading_day"].to_numpy()
    print(f"candidates={n} trading_days={len(np.unique(blocks))}")

    piv = context.pivot(
        index="candidate_id",
        columns="context_tf",
        values=["internal_bias", "swing_bias", "dsa_direction"],
    )
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reindex(
        candidates["candidate_id"].to_numpy()
    ).reset_index(drop=True)
    for c in piv.columns:
        candidates[c] = piv[c].to_numpy()

    arr = build_path_arrays(candidates, path)
    contig = arr["contig"]
    bias = candidates["source_ob_bias"].to_numpy(float)
    rots = {
        "follow": rotated(arr, bias),
        "fade": rotated(arr, -bias),
    }

    idx_symbol = {
        s: np.flatnonzero(
            (candidates["symbol"] == s).to_numpy()
        )
        for s in SYMBOLS
    }
    idx_tf = {
        tf: np.flatnonzero(
            (candidates["source_tf"] == tf).to_numpy()
        )
        for tf in PRIMARY_CONTEXT_TFS
    }

    # ---------------- B: ambiguity gate ----------------
    amb_rows = []
    for dname in DIRECTIONS:
        for h in HORIZONS:
            for stop in STOPS_ATR:
                for tr in TARGET_R:
                    base, code = simulate(
                        rots[dname], contig, stop, tr, h, True
                    )
                    analyzed = (
                        code != EXIT_INSUFFICIENT
                    ) & (code != EXIT_NONCONTIG)
                    if analyzed.sum() == 0:
                        continue
                    r_w = weighted_rate(
                        code == EXIT_BOTH, analyzed, w
                    )
                    r_ev = float(
                        (code == EXIT_BOTH)[analyzed].mean()
                    )
                    amb_rows.append(
                        {
                            "symbol": "ALL",
                            "direction": dname,
                            "stop_atr": stop,
                            "target_r": tr,
                            "horizon": h,
                            "n": int(analyzed.sum()),
                            "both_hit_rate_w": round(r_w, 6),
                            "both_hit_rate_event": round(
                                r_ev, 6
                            ),
                            "gap_stop_rate_w": round(
                                weighted_rate(
                                    code == EXIT_GAP_STOP,
                                    analyzed,
                                    w,
                                ),
                                6,
                            ),
                            "granularity_status": (
                                ambiguity_status(r_w)
                            ),
                        }
                    )
    amb = pd.DataFrame(amb_rows)
    amb.to_csv(RESULTS / "rr_ambiguity_gate.csv", index=False)
    print("rr_ambiguity_gate", len(amb), flush=True)

    # ---------------- C: quantile by symbol x fold ----------
    qfold = candidates["quant_fold"].to_numpy(float)
    qbin = candidates["quant_bin"].to_numpy(object)
    sym_arr = candidates["symbol"].to_numpy(object)

    # Simulation depends only on (stop, target) for the fixed
    # direction / horizon -- cache and reuse.
    qsim = {}
    for stop in QUANT_STOPS:
        for tr in QUANT_TARGETS:
            base, code = simulate(
                rots["follow"],
                contig,
                stop,
                tr,
                REF_HORIZON,
                True,
            )
            qsim[(stop, tr)] = (
                apply_policy(
                    base, code, tr, PRIMARY_POLICY
                ),
                code,
            )

    qfold_int = np.where(
        np.isfinite(qfold), qfold, -1.0
    ).astype(int)
    folds_by_symbol = {
        s: sorted(
            set(
                qfold_int[
                    (sym_arr == s) & (qfold_int >= 0)
                ].tolist()
            )
        )
        for s in SYMBOLS
    }

    q_rows = []
    for symbol in SYMBOLS:
        for fold in folds_by_symbol[symbol]:
            sel_fold = (sym_arr == symbol) & (
                qfold_int == fold
            )
            for stop in QUANT_STOPS:
                for tr in QUANT_TARGETS:
                    res, code = qsim[(stop, tr)]
                    for b in QBINS:
                        sel = sel_fold & (qbin == b)
                        if sel.sum() == 0:
                            continue
                        st = agg_rr_weighted(
                            res,
                            code,
                            w,
                            np.flatnonzero(sel),
                        )
                        if st is None:
                            continue
                        q_rows.append(
                            {
                                "symbol": symbol,
                                "fold": fold,
                                "stop_atr": stop,
                                "target_r": tr,
                                "quant_bin": b,
                                **st,
                            }
                        )
    qdf = pd.DataFrame(q_rows)
    qdf.to_csv(
        RESULTS / "quantile_by_symbol_fold.csv", index=False
    )
    print("quantile_by_symbol_fold", len(qdf), flush=True)

    # stability: per (symbol, fold, stop, target) delta HIGH-LOW
    stab_rows = []
    for symbol in SYMBOLS:
        for fold in (
            list(folds_by_symbol[symbol]) + ["ALL"]
        ):
            for stop in QUANT_STOPS:
                for tr in QUANT_TARGETS:
                    res, code = qsim[(stop, tr)]
                    means = {}
                    for b in QBINS:
                        if fold == "ALL":
                            sel = (
                                (sym_arr == symbol)
                                & (qbin == b)
                            )
                        else:
                            sel = (
                                (sym_arr == symbol)
                                & (qfold_int == fold)
                                & (qbin == b)
                            )
                        if sel.sum() == 0:
                            means[b] = np.nan
                            continue
                        st = agg_rr_weighted(
                            res,
                            code,
                            w,
                            np.flatnonzero(sel),
                        )
                        means[b] = (
                            st["mean_R"] if st else np.nan
                        )
                    d = (
                        means["HIGH"] - means["LOW"]
                        if np.isfinite(means.get("HIGH", np.nan))
                        and np.isfinite(means.get("LOW", np.nan))
                        else np.nan
                    )
                    stab_rows.append(
                        {
                            "symbol": symbol,
                            "fold": fold,
                            "stop_atr": stop,
                            "target_r": tr,
                            "mean_R_LOW": means.get("LOW"),
                            "mean_R_MID": means.get("MID"),
                            "mean_R_HIGH": means.get("HIGH"),
                            "delta_high_low": d,
                        }
                    )
    stab = pd.DataFrame(stab_rows)
    stab.to_csv(
        RESULTS / "quantile_stability.csv", index=False
    )
    print("quantile_stability", len(stab), flush=True)

    # ---------------- D/F: source TF stability --------------
    base_ref, code_ref = simulate(
        rots["follow"],
        contig,
        REF_STOP,
        REF_TARGET,
        REF_HORIZON,
        True,
    )
    res_ref = apply_policy(
        base_ref, code_ref, REF_TARGET, PRIMARY_POLICY
    )

    src_rows = []
    boot_rows = []

    for tf in PRIMARY_CONTEXT_TFS:
        sel = idx_tf[tf]
        st = agg_rr_weighted(res_ref, code_ref, w, sel)
        if st is None:
            continue
        bs = block_bootstrap_mean(
            res_ref[sel], w[sel], blocks[sel]
        )
        src_rows.append(
            {
                "scope": "pooled",
                "symbol": "ALL",
                "source_tf": tf,
                "n": st["n"],
                "n_weighted": st["n_weighted"],
                "mean_R": st["mean_R"],
                "mean_R_event": st["mean_R_event"],
                "target_hit_rate_w": st[
                    "target_hit_rate_w"
                ],
                "stop_hit_rate_w": st["stop_hit_rate_w"],
                "ci025": bs["ci025"],
                "ci975": bs["ci975"],
                "p_gt_0": bs["p_gt_0"],
            }
        )
        boot_rows.append(
            {
                "metric": f"source_tf_{tf}",
                "scope": "ALL",
                "n": st["n"],
                **bs,
            }
        )
        for s in SYMBOLS:
            sel2 = np.intersect1d(sel, idx_symbol[s])
            st2 = agg_rr_weighted(
                res_ref, code_ref, w, sel2
            )
            if st2 is None:
                continue
            bs2 = block_bootstrap_mean(
                res_ref[sel2], w[sel2], blocks[sel2]
            )
            src_rows.append(
                {
                    "scope": "symbol",
                    "symbol": s,
                    "source_tf": tf,
                    "n": st2["n"],
                    "n_weighted": st2["n_weighted"],
                    "mean_R": st2["mean_R"],
                    "mean_R_event": st2["mean_R_event"],
                    "target_hit_rate_w": st2[
                        "target_hit_rate_w"
                    ],
                    "stop_hit_rate_w": st2[
                        "stop_hit_rate_w"
                    ],
                    "ci025": bs2["ci025"],
                    "ci975": bs2["ci975"],
                    "p_gt_0": bs2["p_gt_0"],
                }
            )
            boot_rows.append(
                {
                    "metric": f"source_tf_{tf}",
                    "scope": s,
                    "n": st2["n"],
                    **bs2,
                }
            )

    src = pd.DataFrame(src_rows)
    src.to_csv(
        RESULTS / "source_tf_stability.csv", index=False
    )
    print("source_tf_stability", len(src), flush=True)

    # contribution shares for pooled 15m
    shares = {}
    sel15 = idx_tf["15m"]
    tot = float(np.sum(w[sel15]))
    for s in SYMBOLS:
        s2 = np.intersect1d(sel15, idx_symbol[s])
        shares[s] = (
            round(float(np.sum(w[s2])) / tot, 6)
            if tot > 0
            else np.nan
        )

    # ---------------- G: touch behavior stability -----------
    tb_rows = []
    behavior = candidates["touch_behavior"].to_numpy(object)
    for dname in DIRECTIONS:
        base_d, code_d = (
            (base_ref, code_ref)
            if dname == "follow"
            else simulate(
                rots["fade"],
                contig,
                REF_STOP,
                REF_TARGET,
                REF_HORIZON,
                True,
            )
        )
        res_d = apply_policy(
            base_d, code_d, REF_TARGET, PRIMARY_POLICY
        )
        for s in SYMBOLS:
            for tf in PRIMARY_CONTEXT_TFS:
                for b in (
                    "NO_BREACH",
                    "BREACH_RECLAIM",
                    "CLOSE_BEYOND",
                ):
                    sel = np.flatnonzero(
                        (sym_arr == s)
                        & (
                            candidates["source_tf"].to_numpy(
                                object
                            )
                            == tf
                        )
                        & (behavior == b)
                    )
                    if len(sel) == 0:
                        continue
                    st = agg_rr_weighted(
                        res_d, code_d, w, sel
                    )
                    if st is None:
                        continue
                    if len(sel) >= 300:
                        bs = block_bootstrap_mean(
                            res_d[sel],
                            w[sel],
                            blocks[sel],
                            n_boot=1000,
                        )
                    else:
                        bs = {
                            "ci025": np.nan,
                            "ci975": np.nan,
                            "p_gt_0": np.nan,
                        }
                    tb_rows.append(
                        {
                            "symbol": s,
                            "source_tf": tf,
                            "touch_behavior": b,
                            "direction": dname,
                            "n": st["n"],
                            "n_weighted": st["n_weighted"],
                            "mean_R": st["mean_R"],
                            "mean_R_event": st[
                                "mean_R_event"
                            ],
                            "target_hit_rate_w": st[
                                "target_hit_rate_w"
                            ],
                            "ci025": bs["ci025"],
                            "ci975": bs["ci975"],
                            "p_gt_0": bs["p_gt_0"],
                        }
                    )
    tb = pd.DataFrame(tb_rows)
    tb.to_csv(
        RESULTS / "touch_behavior_stability.csv", index=False
    )
    print("touch_behavior_stability", len(tb), flush=True)

    # ---------------- C bootstrap: quantile bins ------------
    qboot_rows = []
    for scope in ("ALL",) + SYMBOLS:
        sel_scope = (
            np.ones(n, dtype=bool)
            if scope == "ALL"
            else (sym_arr == scope)
        )
        sel = (
            sel_scope
            & np.isfinite(qfold)
            & np.isin(qbin, ["HIGH", "LOW"])
        )
        if sel.sum() < 50:
            continue
        vals = res_ref[sel]
        ww = w[sel]
        bb = blocks[sel]
        gg = qbin[sel]
        gb = block_bootstrap_grouped(
            vals, ww, bb, gg
        )
        for gname, stt in gb.items():
            qboot_rows.append(
                {
                    "metric": (
                        "quantile_bin"
                        if gname in QBINS
                        else "quantile_delta"
                    ),
                    "scope": scope,
                    "group": gname,
                    "n": int(sel.sum()),
                    "mean": stt.get("mean"),
                    "ci025": stt.get("ci025"),
                    "ci975": stt.get("ci975"),
                    "p_gt_0": stt.get("p_gt_0"),
                }
            )

    # ---------------- H: DSA purged + weighted -------------
    dsa_rows = []
    for dname in DIRECTIONS:
        dir_arr = bias if dname == "follow" else -bias
        feat_s = []
        feat_d = []
        for tf in PRIMARY_CONTEXT_TFS:
            for kind in ("internal_bias", "swing_bias"):
                v = candidates[f"{kind}_{tf}"].to_numpy(
                    float
                )
                feat_s.append(v * dir_arr)
        for tf in PRIMARY_CONTEXT_TFS:
            v = candidates[f"dsa_direction_{tf}"].to_numpy(
                float
            )
            feat_d.append(v * dir_arr)

        X_s = np.column_stack(feat_s)
        X_d = np.column_stack(feat_s + feat_d)

        base_d, code_d = simulate(
            rots[dname],
            contig,
            REF_STOP,
            REF_TARGET,
            REF_HORIZON,
            True,
        )
        res_d = apply_policy(
            base_d, code_d, REF_TARGET, PRIMARY_POLICY
        )
        analyzed = (code_d != EXIT_INSUFFICIENT) & (
            code_d != EXIT_NONCONTIG
        )
        y_tgt = (
            (code_d == EXIT_TARGET)
            | (code_d == EXIT_GAP_TARGET)
        ).astype(float)
        y_pos = np.where(
            np.isfinite(res_d),
            (res_d > 0).astype(float),
            np.nan,
        )

        for scope in ("ALL",) + SYMBOLS:
            scope_mask = (
                np.ones(n, dtype=bool)
                if scope == "ALL"
                else (sym_arr == scope)
            )
            for oname, y in (
                ("target_hit", y_tgt),
                ("positive_R", y_pos),
            ):
                ok = (
                    analyzed
                    & scope_mask
                    & np.isfinite(y)
                    & np.isfinite(X_s).all(axis=1)
                    & np.isfinite(X_d).all(axis=1)
                )
                if ok.sum() < 500:
                    continue
                oof = {"S": [], "S+D": []}
                ys, ws = [], []
                for _, tr, te in purged_expanding_splits(
                    candidates, n_folds=5, embargo_minutes=60
                ):
                    tr = tr[ok[tr]]
                    te = te[ok[te]]
                    if len(tr) < 200 or len(te) < 50:
                        continue
                    for name, X in (
                        ("S", X_s),
                        ("S+D", X_d),
                    ):
                        m = LogisticRegression(
                            C=1.0,
                            penalty="l2",
                            max_iter=2000,
                        )
                        m.fit(
                            X[tr],
                            y[tr],
                            sample_weight=w[tr],
                        )
                        oof[name].append(
                            m.predict_proba(X[te])[:, 1]
                        )
                        if name == "S":
                            ys.append(y[te])
                            ws.append(w[te])
                if not ys:
                    continue
                yv = np.concatenate(ys)
                wv = np.concatenate(ws)
                for name in ("S", "S+D"):
                    pv = np.concatenate(oof[name])
                    if len(np.unique(yv)) < 2:
                        auc = np.nan
                    else:
                        auc = float(
                            roc_auc_score(
                                yv, pv, sample_weight=wv
                            )
                        )
                    dsa_rows.append(
                        {
                            "symbol": scope,
                            "direction": dname,
                            "outcome": (
                                f"1ATR_2R_H12_{oname}"
                            ),
                            "model": name,
                            "n_oos": int(len(yv)),
                            "auc_w": round(auc, 6),
                            "brier_w": round(
                                float(
                                    brier_score_loss(
                                        yv,
                                        pv,
                                        sample_weight=wv,
                                    )
                                ),
                                6,
                            ),
                            "logloss_w": round(
                                float(
                                    log_loss(
                                        yv,
                                        pv,
                                        sample_weight=wv,
                                    )
                                ),
                                6,
                            ),
                        }
                    )
    dsa = pd.DataFrame(dsa_rows)
    dsa.to_csv(
        RESULTS / "dsa_purged_increment.csv", index=False
    )
    print("dsa_purged_increment", len(dsa), flush=True)

    pd.DataFrame(boot_rows + qboot_rows).to_csv(
        RESULTS / "bootstrap_ci.csv", index=False
    )

    # ---------------- summary ----------------
    max_both = float(amb["both_hit_rate_w"].max())
    gt2 = int((amb["both_hit_rate_w"] > 0.02).sum())
    gt10 = int((amb["both_hit_rate_w"] > 0.10).sum())

    cells = stab.dropna(subset=["delta_high_low"])
    cells_fold = cells[cells["fold"].astype(str) != "ALL"]
    frac_all = (
        float((cells_fold["delta_high_low"] > 0).mean())
        if len(cells_fold)
        else np.nan
    )
    frac_by_symbol = {}
    for s in SYMBOLS:
        sub = cells_fold[cells_fold["symbol"] == s]
        frac_by_symbol[s] = (
            round(
                float((sub["delta_high_low"] > 0).mean()), 6
            )
            if len(sub)
            else np.nan
        )
    sym_pos = sum(
        1
        for s in SYMBOLS
        if np.isfinite(frac_by_symbol.get(s, np.nan))
        and frac_by_symbol[s] > 0.5
    )

    tf15 = src[
        (src.source_tf == "15m") & (src.scope == "symbol")
    ]
    tf15_pos = int((tf15["mean_R"] > 0).sum())

    summary = {
        "schema_version": "ob_candidate_v3_phase1_1",
        "parent_sha": (
            "f84a3045dee89619a433ec99b4a2916d2435bcaa"
        ),
        "candidates": int(n),
        "trading_days": int(len(np.unique(blocks))),
        "weighting": {
            "primary": "decision_bar_weighted",
            "all_rates_weighted": True,
            "event_kept_as_robustness": True,
        },
        "ambiguity": {
            "max_both_hit_rate": round(max_both, 6),
            "configs_gt_2pct": gt2,
            "configs_gt_10pct": gt10,
            "total_configs": int(len(amb)),
            "note": (
                "per-configuration gate; primary grid must use "
                "5m_OK configs only"
            ),
        },
        "source_tf": {
            "15m_symbols_positive": tf15_pos,
            "15m_contribution_shares": shares,
        },
        "quantile": {
            "coverage_n": int(np.isfinite(qfold).sum()),
            "coverage_rate": round(
                float(np.isfinite(qfold).mean()), 6
            ),
            "cells": int(len(cells_fold)),
            "fraction_delta_gt_0": (
                round(frac_all, 6)
                if np.isfinite(frac_all)
                else None
            ),
            "fraction_delta_gt_0_by_symbol": (
                frac_by_symbol
            ),
            "symbols_with_majority_positive": sym_pos,
            "pass": bool(
                np.isfinite(frac_all)
                and frac_all >= 0.60
                and sym_pos >= 3
            ),
        },
        "dsa": {
            "rows": int(len(dsa)),
        },
        "confirm": {
            "v3_changed": 0,
            "phase1_changed": 0,
        },
    }
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("PHASE1_1_DONE", flush=True)


if __name__ == "__main__":
    main()
