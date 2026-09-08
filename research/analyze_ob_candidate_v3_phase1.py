#!/usr/bin/env python3

"""Phase E-M discovery analysis for OB Candidate Universe V3.

Reads the FROZEN V3 dataset and the post-hoc annotations, then runs the
pre-registered Phase-1 discovery battery:

    E  4h quarantine (environment only, never a feature)
    F  Quantile re-validation on ATR / return scale
    G  Directional MFE / MAE (FOLLOW and FADE)
    H  RR simulator (fixed grid, 3 both-hit policies)
    I  Single-factor cuts (source TF, touch ordinal, touch behavior,
       internal/swing, confluence)
    J  SMC vs DSA (conditional table + nested logistic model)
    K  Quantile opportunity (LOW / MID / HIGH)
    L  Event-weight vs decision-bar-weight
    M  AG / CU / RB / M only

No strategy optimisation is performed: the stop / target grid is fixed.
No SL / TP / PnL field is written back into the frozen V3 dataset.
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

RESULTS = ROOT / "research" / "analysis_results" / "ob_candidate_v3_phase1"

from research.audit_ob_candidate_v3_posthoc import (  # noqa: E402
    load_full_or_chunks,
    add_true_continuity,
)

# Phase E - 4h is quarantined.
PRIMARY_CONTEXT_TFS = ("5m", "15m", "1h")
QUARANTINED_TFS = ("4h",)

SYMBOLS = ("AG", "CU", "RB", "M")
STOPS_ATR = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
TARGET_R = (1.0, 1.5, 2.0, 2.5, 3.0)
HORIZONS = (6, 12, 24)
POLICIES = ("conservative", "exclude", "optimistic")
PRIMARY_POLICY = "conservative"
DIRECTIONS = ("follow", "fade")

REF = {
    "stop_atr": 1.00,
    "target_r": 2.0,
    "horizon": 12,
}

EXIT_INSUFFICIENT = 0
EXIT_NONCONTIG = 1
EXIT_GAP_STOP = 2
EXIT_GAP_TARGET = 3
EXIT_BOTH = 4
EXIT_STOP = 5
EXIT_TARGET = 6
EXIT_TIMEOUT = 7


# ============================================================
# Weighted statistics (Phase L)
# ============================================================

def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(w)
    if not m.any():
        return float("nan")
    denom = float(np.sum(w[m]))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x[m] * w[m]) / denom)


def weighted_quantile(
    x: np.ndarray,
    q: float,
    w: np.ndarray,
) -> float:
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not m.any():
        return float("nan")
    xv = x[m]
    wv = w[m]
    o = np.argsort(xv)
    xv = xv[o]
    wv = wv[o]
    cw = np.cumsum(wv) - 0.5 * wv
    cw /= np.sum(wv)
    return float(np.interp(q, cw, xv))


def agg_rr(
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

    def rate(mask: np.ndarray) -> float:
        return float(mask[analyzed].mean())

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
                "mean_R": np.nan,
                "mean_R_event": np.nan,
                "median_R": np.nan,
                "positive_R_rate": np.nan,
                "p10_R": np.nan,
                "p90_R": np.nan,
            }
        )
    else:
        r = result[valid]
        ww = w[valid]
        out.update(
            {
                "mean_R": round(weighted_mean(r, ww), 6),
                "mean_R_event": round(float(np.mean(r)), 6),
                "median_R": round(float(np.median(r)), 6),
                "positive_R_rate": round(
                    float(np.mean(r > 0)), 6
                ),
                "p10_R": round(
                    weighted_quantile(r, 0.10, ww), 6
                ),
                "p90_R": round(
                    weighted_quantile(r, 0.90, ww), 6
                ),
            }
        )

    out.update(
        {
            "target_hit_rate": round(
                rate(
                    (code == EXIT_TARGET)
                    | (code == EXIT_GAP_TARGET)
                ),
                6,
            ),
            "stop_hit_rate": round(
                rate(
                    (code == EXIT_STOP)
                    | (code == EXIT_GAP_STOP)
                ),
                6,
            ),
            "timeout_rate": round(
                rate(code == EXIT_TIMEOUT), 6
            ),
            "both_hit_rate": round(
                rate(code == EXIT_BOTH), 6
            ),
            "gap_stop_rate": round(
                rate(code == EXIT_GAP_STOP), 6
            ),
            "gap_target_rate": round(
                rate(code == EXIT_GAP_TARGET), 6
            ),
        }
    )
    return out


# ============================================================
# Path arrays
# ============================================================

def build_path_arrays(
    candidates: pd.DataFrame,
    path: pd.DataFrame,
) -> dict:
    path = add_true_continuity(path)
    piv = path.pivot(
        index="candidate_id",
        columns="step",
        values=[
            "open_atr",
            "high_atr",
            "low_atr",
            "close_atr",
            "execution_contiguous",
        ],
    )
    piv.columns = [
        f"{a}_{b}" for a, b in piv.columns
    ]
    piv = piv.reindex(candidates["candidate_id"].to_numpy())

    def grid(name: str) -> np.ndarray:
        cols = [
            c
            for c in piv.columns
            if c.startswith(name + "_")
        ]
        # Numeric step ordering (string sort puts _10 before _2).
        cols = sorted(
            cols, key=lambda c: int(c.rsplit("_", 1)[1])
        )
        return piv[cols].to_numpy(float)

    return {
        "open_atr": grid("open_atr"),
        "high_atr": grid("high_atr"),
        "low_atr": grid("low_atr"),
        "close_atr": grid("close_atr"),
        "contig": grid("execution_contiguous").astype(bool),
    }


def rotated(
    arr: dict,
    sign: np.ndarray,
) -> dict:
    """Rotate coordinates so that + means trade-favourable."""
    s = sign[:, None]
    o = arr["open_atr"]
    hi = arr["high_atr"]
    lo = arr["low_atr"]
    cl = arr["close_atr"]
    return {
        "o": o * s,
        "hi": np.where(s == 1, hi, -lo),
        "lo": np.where(s == 1, lo, -hi),
        "cl": cl * s,
    }


# ============================================================
# Phase H - vectorised first-hit RR simulator
# ============================================================

def simulate(
    rot: dict,
    contig: np.ndarray,
    stop_atr: float,
    target_r: float,
    horizon: int,
    require_contiguous: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    o = rot["o"][:, :horizon]
    hi = rot["hi"][:, :horizon]
    lo = rot["lo"][:, :horizon]
    cl = rot["cl"][:, :horizon]
    ct = contig[:, :horizon]

    n = o.shape[0]
    finite = (
        np.isfinite(o)
        & np.isfinite(hi)
        & np.isfinite(lo)
        & np.isfinite(cl)
    )
    sufficient = finite.all(axis=1)
    contig_ok = ct.all(axis=1)

    if require_contiguous:
        analyzable = sufficient & contig_ok
    else:
        analyzable = sufficient

    code = np.full(n, EXIT_INSUFFICIENT, dtype=int)
    if require_contiguous:
        code = np.where(
            sufficient & ~contig_ok,
            EXIT_NONCONTIG,
            code,
        )

    result = np.full(n, np.nan)
    if not analyzable.any():
        return result, code

    target_atr = stop_atr * target_r

    gap_stop = o <= -stop_atr
    gap_target = o >= target_atr
    hit_stop = lo <= -stop_atr
    hit_target = hi >= target_atr
    event = gap_stop | gap_target | hit_stop | hit_target

    ev = np.where(analyzable[:, None], event, False)
    any_ev = ev.any(axis=1)

    first = np.argmax(ev, axis=1)
    idx = first[:, None]

    def take(a):
        return np.take_along_axis(a, idx, axis=1).ravel()

    o_f = take(o)
    gs = take(gap_stop) & any_ev
    gt = take(gap_target) & any_ev & ~gs
    hs = take(hit_stop) & any_ev
    ht = take(hit_target) & any_ev
    is_both = hs & ht & ~gs & ~gt
    is_stop = hs & ~ht & ~gs & ~gt
    is_target = ht & ~hs & ~gs & ~gt
    is_timeout = analyzable & ~any_ev

    code = np.where(is_timeout, EXIT_TIMEOUT, code)
    code = np.where(gs, EXIT_GAP_STOP, code)
    code = np.where(gt, EXIT_GAP_TARGET, code)
    code = np.where(is_both, EXIT_BOTH, code)
    code = np.where(is_stop, EXIT_STOP, code)
    code = np.where(is_target, EXIT_TARGET, code)
    if require_contiguous:
        code = np.where(
            sufficient & ~contig_ok, EXIT_NONCONTIG, code
        )

    base = np.full(n, np.nan)
    base = np.where(gs | gt, o_f / stop_atr, base)
    base = np.where(is_stop, -1.0, base)
    base = np.where(is_target, target_r, base)
    base = np.where(is_timeout, cl[:, -1] / stop_atr, base)
    return base, code


def apply_policy(
    base: np.ndarray,
    code: np.ndarray,
    target_r: float,
    policy: str,
) -> np.ndarray:
    if policy == "conservative":
        return np.where(code == EXIT_BOTH, -1.0, base)
    if policy == "optimistic":
        return np.where(code == EXIT_BOTH, target_r, base)
    if policy == "exclude":
        return np.where(code == EXIT_BOTH, np.nan, base)
    raise ValueError(policy)


# ============================================================
# Spearman
# ============================================================

def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a = a[m]
    b = b[m]
    if len(a) < 10:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    da = ra - ra.mean()
    db = rb - rb.mean()
    denom = np.sqrt((da**2).sum() * (db**2).sum())
    if denom == 0:
        return float("nan")
    return float((da * db).sum() / denom)


# ============================================================
# Main
# ============================================================

def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    candidates = load_full_or_chunks("candidates")
    path = load_full_or_chunks("path")
    context = load_full_or_chunks("context")
    ann = pd.read_csv(
        RESULTS / "candidate_annotations.csv"
    )

    candidates = candidates.merge(
        ann,
        on=["candidate_id", "symbol", "source_tf"],
        how="left",
        validate="one_to_one",
    )

    n = len(candidates)
    print(f"candidates={n}", flush=True)

    # Phase L weights.
    w = (
        1.0
        / candidates["group_candidate_count"].to_numpy(float)
    )
    w_event = np.ones(n)

    # Phase E - pivot context (4h present but excluded).
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

    # Path arrays.
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
    idx_all = np.arange(n)
    idx_tf = {
        tf: np.flatnonzero(
            (candidates["source_tf"] == tf).to_numpy()
        )
        for tf in PRIMARY_CONTEXT_TFS
    }
    idx_quant = {
        b: np.flatnonzero(
            (candidates["quant_bin"] == b).to_numpy()
        )
        for b in ("LOW", "MID", "HIGH")
    }
    idx_behavior = {
        b: np.flatnonzero(
            (candidates["touch_behavior"] == b).to_numpy()
        )
        for b in ("NO_BREACH", "BREACH_RECLAIM", "CLOSE_BEYOND")
    }
    # ---------------- Phase G: excursion ----------------
    exc_rows = []
    for dname in DIRECTIONS:
        rot = rots[dname]
        for h in HORIZONS:
            hi = rot["hi"][:, :h]
            lo = rot["lo"][:, :h]
            ct = contig[:, :h]
            ok = (
                np.isfinite(hi).all(axis=1)
                & np.isfinite(lo).all(axis=1)
                & ct.all(axis=1)
            )
            mfe = np.maximum(
                np.max(hi, axis=1), 0.0
            )
            mae = np.maximum(-np.min(lo, axis=1), 0.0)

            def add_exc(label, sel):
                s = sel & ok
                if s.sum() == 0:
                    return
                ww = w[s]
                mm = mfe[s]
                aa = mae[s]
                mfe_m = weighted_mean(mm, ww)
                mae_m = weighted_mean(aa, ww)
                exc_rows.append(
                    {
                        **label,
                        "direction": dname,
                        "horizon": h,
                        "n": int(s.sum()),
                        "n_weighted": round(
                            float(ww.sum()), 4
                        ),
                        "mean_mfe_atr": round(mfe_m, 6),
                        "mean_mae_atr": round(mae_m, 6),
                        "mean_net_atr": round(
                            mfe_m - mae_m, 6
                        ),
                        "mfe_mae_ratio": (
                            round(mfe_m / mae_m, 6)
                            if mae_m > 0
                            else np.nan
                        ),
                    }
                )

            for s in SYMBOLS:
                add_exc(
                    {
                        "symbol": s,
                        "source_tf": "ALL",
                    },
                    (candidates["symbol"] == s).to_numpy(),
                )
            for tf in PRIMARY_CONTEXT_TFS:
                add_exc(
                    {
                        "symbol": "ALL",
                        "source_tf": tf,
                    },
                    (candidates["source_tf"] == tf).to_numpy(),
                )

    pd.DataFrame(exc_rows).to_csv(
        RESULTS / "excursion_by_horizon.csv", index=False
    )
    print("excursion_by_horizon.csv", len(exc_rows), flush=True)

    # ---------------- Phase H: RR grid ----------------
    rr_rows = []
    cut_rows = {
        "source_tf": [],
        "quantile": [],
        "touch_state": [],
    }

    for dname in DIRECTIONS:
        rot = rots[dname]
        for h in HORIZONS:
            for stop in STOPS_ATR:
                for tr in TARGET_R:
                    base, code = simulate(
                        rot, contig, stop, tr, h, True
                    )
                    for policy in POLICIES:
                        res = apply_policy(
                            base, code, tr, policy
                        )

                        def emit(symbol, sel):
                            st = agg_rr(res, code, w, sel)
                            if st is None:
                                return
                            rr_rows.append(
                                {
                                    "symbol": symbol,
                                    "direction": dname,
                                    "policy": policy,
                                    "stop_atr": stop,
                                    "target_r": tr,
                                    "horizon": h,
                                    **st,
                                }
                            )

                        for s in SYMBOLS:
                            emit(s, idx_symbol[s])
                        emit("ALL", idx_all)

                        if policy == PRIMARY_POLICY:
                            for tf in PRIMARY_CONTEXT_TFS:
                                cut_rows["source_tf"].append(
                                    {
                                        "symbol": "ALL",
                                        "source_tf": tf,
                                        "direction": dname,
                                        "policy": policy,
                                        "stop_atr": stop,
                                        "target_r": tr,
                                        "horizon": h,
                                        **(
                                            agg_rr(
                                                res,
                                                code,
                                                w,
                                                idx_tf[tf],
                                            )
                                            or {}
                                        ),
                                    }
                                )
                            for b, sel in idx_quant.items():
                                cut_rows["quantile"].append(
                                    {
                                        "symbol": "ALL",
                                        "quant_bin": b,
                                        "direction": dname,
                                        "policy": policy,
                                        "stop_atr": stop,
                                        "target_r": tr,
                                        "horizon": h,
                                        **(
                                            agg_rr(
                                                res, code, w, sel
                                            )
                                            or {}
                                        ),
                                    }
                                )
                            for b, sel in idx_behavior.items():
                                cut_rows["touch_state"].append(
                                    {
                                        "symbol": "ALL",
                                        "touch_behavior": b,
                                        "direction": dname,
                                        "policy": policy,
                                        "stop_atr": stop,
                                        "target_r": tr,
                                        "horizon": h,
                                        **(
                                            agg_rr(
                                                res, code, w, sel
                                            )
                                            or {}
                                        ),
                                    }
                                )

    pd.DataFrame(rr_rows).to_csv(
        RESULTS / "rr_grid.csv", index=False
    )
    pd.DataFrame(cut_rows["source_tf"]).to_csv(
        RESULTS / "rr_by_source_tf.csv", index=False
    )
    pd.DataFrame(cut_rows["quantile"]).to_csv(
        RESULTS / "rr_by_quantile.csv", index=False
    )
    pd.DataFrame(cut_rows["touch_state"]).to_csv(
        RESULTS / "rr_by_touch_state.csv", index=False
    )
    print(
        "rr_grid",
        len(rr_rows),
        "by_tf",
        len(cut_rows["source_tf"]),
        flush=True,
    )

    # ---------------- Reference stats for cuts ----------------
    ref_cache = {}
    for dname in DIRECTIONS:
        base, code = simulate(
            rots[dname],
            contig,
            REF["stop_atr"],
            REF["target_r"],
            REF["horizon"],
            True,
        )
        ref_cache[dname] = (
            apply_policy(
                base, code, REF["target_r"], PRIMARY_POLICY
            ),
            code,
        )

    # baseline_by_source_tf
    rows = []
    for s in SYMBOLS:
        for tf in PRIMARY_CONTEXT_TFS:
            sel = (
                (candidates["symbol"] == s)
                & (candidates["source_tf"] == tf)
            ).to_numpy()
            if sel.sum() == 0:
                continue
            sub = candidates[sel]
            st_f = agg_rr(
                ref_cache["follow"][0], ref_cache["follow"][1], w, np.flatnonzero(sel)
            )
            st_d = agg_rr(
                ref_cache["fade"][0], ref_cache["fade"][1], w, np.flatnonzero(sel)
            )
            rows.append(
                {
                    "symbol": s,
                    "source_tf": tf,
                    "n": int(sel.sum()),
                    "n_weighted": round(
                        float(w[sel].sum()), 4
                    ),
                    "first_touch": int(
                        sub["is_first_touch"].sum()
                    ),
                    "retest": int(
                        (~sub["is_first_touch"]).sum()
                    ),
                    "internal": int(
                        sub["source_ob_internal"].sum()
                    ),
                    "swing": int(
                        (~sub["source_ob_internal"]).sum()
                    ),
                    "bull": int(
                        (sub["source_ob_bias"] == 1).sum()
                    ),
                    "bear": int(
                        (sub["source_ob_bias"] == -1).sum()
                    ),
                    "active_at_touch": int(
                        sub[
                            "source_ob_active_at_touch_close"
                        ].sum()
                    ),
                    "mitigated_at_touch": int(
                        sub[
                            "source_ob_mitigated_at_touch_close"
                        ].sum()
                    ),
                    "mean_R_follow": (
                        st_f["mean_R"] if st_f else np.nan
                    ),
                    "mean_R_fade": (
                        st_d["mean_R"] if st_d else np.nan
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(
        RESULTS / "baseline_by_source_tf.csv", index=False
    )

    # first_retest / touch_state
    for fname, col, order in (
        (
            "first_retest.csv",
            "touch_bin",
            ("1", "2", "3", "4+"),
        ),
        (
            "touch_state.csv",
            "touch_behavior",
            ("NO_BREACH", "BREACH_RECLAIM", "CLOSE_BEYOND"),
        ),
    ):
        rows = []
        for s in SYMBOLS:
            for v in order:
                sel = (
                    (candidates["symbol"] == s)
                    & (candidates[col] == v)
                ).to_numpy()
                if sel.sum() == 0:
                    continue
                i = np.flatnonzero(sel)
                st_f = agg_rr(ref_cache["follow"][0], ref_cache["follow"][1], w, i)
                st_d = agg_rr(ref_cache["fade"][0], ref_cache["fade"][1], w, i)
                rows.append(
                    {
                        "symbol": s,
                        col: v,
                        "n": int(sel.sum()),
                        "n_weighted": round(
                            float(w[sel].sum()), 4
                        ),
                        "mean_R_follow": (
                            st_f["mean_R"] if st_f else np.nan
                        ),
                        "mean_R_fade": (
                            st_d["mean_R"] if st_d else np.nan
                        ),
                        "target_hit_follow": (
                            st_f["target_hit_rate"]
                            if st_f
                            else np.nan
                        ),
                    }
                )
        pd.DataFrame(rows).to_csv(
            RESULTS / fname, index=False
        )

    # confluence
    conf_order = (
        "only_5m",
        "only_15m",
        "only_1h",
        "5m+15m",
        "5m+1h",
        "15m+1h",
        "5m+15m+1h",
    )
    has = {
        "5m": candidates["group_has_5m"].to_numpy(bool),
        "15m": candidates["group_has_15m"].to_numpy(bool),
        "1h": candidates["group_has_1h"].to_numpy(bool),
    }
    conf = np.full(n, "other", dtype=object)
    conf[
        has["5m"] & ~has["15m"] & ~has["1h"]
    ] = "only_5m"
    conf[
        ~has["5m"] & has["15m"] & ~has["1h"]
    ] = "only_15m"
    conf[
        ~has["5m"] & ~has["15m"] & has["1h"]
    ] = "only_1h"
    conf[
        has["5m"] & has["15m"] & ~has["1h"]
    ] = "5m+15m"
    conf[
        has["5m"] & ~has["15m"] & has["1h"]
    ] = "5m+1h"
    conf[
        ~has["5m"] & has["15m"] & has["1h"]
    ] = "15m+1h"
    conf[
        has["5m"] & has["15m"] & has["1h"]
    ] = "5m+15m+1h"
    conflict = candidates["group_bias_conflict"].to_numpy(bool)

    rows = []
    for s in SYMBOLS:
        for c in conf_order:
            for rel in ("same_bias", "bias_conflict"):
                sel = (
                    (candidates["symbol"] == s)
                    & (conf == c)
                    & (
                        conflict
                        if rel == "bias_conflict"
                        else ~conflict
                    )
                )
                if sel.sum() == 0:
                    continue
                i = np.flatnonzero(sel)
                st_f = agg_rr(ref_cache["follow"][0], ref_cache["follow"][1], w, i)
                st_d = agg_rr(ref_cache["fade"][0], ref_cache["fade"][1], w, i)
                rows.append(
                    {
                        "symbol": s,
                        "confluence": c,
                        "bias_relation": rel,
                        "n": int(sel.sum()),
                        "n_weighted": round(
                            float(w[sel].sum()), 4
                        ),
                        "mean_R_follow": (
                            st_f["mean_R"] if st_f else np.nan
                        ),
                        "mean_R_fade": (
                            st_d["mean_R"] if st_d else np.nan
                        ),
                    }
                )
    pd.DataFrame(rows).to_csv(
        RESULTS / "confluence.csv", index=False
    )
    print("confluence.csv", len(rows), flush=True)

    # ---------------- Phase J: SMC vs DSA ----------------
    def rel_state(col_tf, direction_arr):
        v = candidates[col_tf].to_numpy(float)
        return v * direction_arr

    align_rows = []
    incr_rows = []
    for dname in DIRECTIONS:
        dir_arr = (
            bias if dname == "follow" else -bias
        )
        smc_rel = {}
        for tf in PRIMARY_CONTEXT_TFS:
            smc_rel[tf] = rel_state(
                f"internal_bias_{tf}", dir_arr
            )
        smc_count = np.sum(
            [
                (smc_rel[tf] == 1).astype(int)
                for tf in PRIMARY_CONTEXT_TFS
            ],
            axis=0,
        )
        dsa_rel = {}
        for tf in PRIMARY_CONTEXT_TFS:
            dsa_rel[tf] = rel_state(
                f"dsa_direction_{tf}", dir_arr
            )

        for lvl in range(0, 4):
            lvl_sel = smc_count == lvl
            for tf in PRIMARY_CONTEXT_TFS:
                d = dsa_rel[tf]
                for state, mask_v in (
                    ("aligned", 1),
                    ("flat", 0),
                    ("opposed", -1),
                ):
                    sel = lvl_sel & (d == mask_v)
                    if sel.sum() == 0:
                        continue
                    i = np.flatnonzero(sel)
                    st = agg_rr(
                        ref_cache[dname][0],
                        ref_cache[dname][1],
                        w,
                        i,
                    )
                    if st is None:
                        continue
                    align_rows.append(
                        {
                            "symbol": "ALL",
                            "direction": dname,
                            "smc_internal_align_count": lvl,
                            "dsa_tf": tf,
                            "dsa_state": state,
                            "n": st["n"],
                            "n_weighted": st["n_weighted"],
                            "mean_R": st["mean_R"],
                            "target_hit_rate": st[
                                "target_hit_rate"
                            ],
                            "positive_R_rate": st[
                                "positive_R_rate"
                            ],
                        }
                    )

                vals = {}
                for state, mask_v in (
                    ("aligned", 1),
                    ("flat", 0),
                    ("opposed", -1),
                ):
                    sel = lvl_sel & (d == mask_v)
                    st = (
                        agg_rr(
                            ref_cache[dname][0],
                            ref_cache[dname][1],
                            w,
                            np.flatnonzero(sel),
                        )
                        if sel.any()
                        else None
                    )
                    vals[state] = (
                        st["mean_R"] if st else np.nan
                    )
                    vals[state + "_n"] = (
                        st["n"] if st else 0
                    )
                incr_rows.append(
                    {
                        "symbol": "ALL",
                        "direction": dname,
                        "smc_internal_align_count": lvl,
                        "dsa_tf": tf,
                        **vals,
                        "delta_aligned_minus_opposed": (
                            round(
                                vals["aligned"]
                                - vals["opposed"],
                                6,
                            )
                            if np.isfinite(vals["aligned"])
                            and np.isfinite(vals["opposed"])
                            else np.nan
                        ),
                    }
                )

    pd.DataFrame(align_rows).to_csv(
        RESULTS / "smc_alignment.csv", index=False
    )
    pd.DataFrame(incr_rows).to_csv(
        RESULTS / "dsa_increment.csv", index=False
    )
    print(
        "smc_alignment",
        len(align_rows),
        "dsa_increment",
        len(incr_rows),
        flush=True,
    )

    # ---------------- Phase J secondary: nested model --------
    model_rows = []
    outcomes = (
        {"name": "1ATR_1.5R_H12", "target_r": 1.5},
        {"name": "1ATR_2.0R_H12", "target_r": 2.0},
    )
    for dname in DIRECTIONS:
        dir_arr = bias if dname == "follow" else -bias
        feat_s = []
        feat_d = []
        for tf in PRIMARY_CONTEXT_TFS:
            for kind in ("internal_bias", "swing_bias"):
                feat_s.append(
                    rel_state(f"{kind}_{tf}", dir_arr)
                )
        for tf in PRIMARY_CONTEXT_TFS:
            feat_d.append(
                rel_state(f"dsa_direction_{tf}", dir_arr)
            )

        X_s = np.column_stack(feat_s)
        X_d = np.column_stack(feat_s + feat_d)

        for oc in outcomes:
            base, code = simulate(
                rots[dname],
                contig,
                REF["stop_atr"],
                oc["target_r"],
                REF["horizon"],
                True,
            )
            res = apply_policy(
                base, code, oc["target_r"], PRIMARY_POLICY
            )
            y = np.where(
                np.isfinite(res), (res > 0).astype(float), np.nan
            )

            ok = (
                np.isfinite(X_s).all(axis=1)
                & np.isfinite(X_d).all(axis=1)
                & np.isfinite(y)
                & (code != EXIT_INSUFFICIENT)
                & (code != EXIT_NONCONTIG)
            )

            if ok.sum() < 500:
                continue

            tt = pd.to_datetime(
                candidates["touch_time"]
            ).to_numpy("datetime64[ns]")
            for scope in ("ALL",) + SYMBOLS:
                scope_mask = (
                    np.ones(n, dtype=bool)
                    if scope == "ALL"
                    else (
                        candidates["symbol"] == scope
                    ).to_numpy()
                )
                sel = ok & scope_mask
                if sel.sum() < 500:
                    continue
                i = np.flatnonzero(sel)
                order = i[np.argsort(tt[i])]
                k = 5
                chunks = np.array_split(order, k)
                oof = {
                    "S": [],
                    "S+D": [],
                }
                ys = []
                for f in range(1, k):
                    tr = np.concatenate(chunks[:f])
                    te = chunks[f]
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
                        m.fit(X[tr], y[tr])
                        p = m.predict_proba(X[te])[:, 1]
                        oof[name].append(p)
                        if name == "S":
                            ys.append(y[te])
                if not ys:
                    continue
                yv = np.concatenate(ys)
                for name in ("S", "S+D"):
                    pv = np.concatenate(oof[name])
                    if len(np.unique(yv)) < 2:
                        auc = np.nan
                    else:
                        auc = float(
                            roc_auc_score(yv, pv)
                        )
                    model_rows.append(
                        {
                            "symbol": scope,
                            "direction": dname,
                            "outcome": oc["name"],
                            "model": name,
                            "n_oos": int(len(yv)),
                            "auc": round(auc, 6),
                            "brier": round(
                                float(
                                    brier_score_loss(yv, pv)
                                ),
                                6,
                            ),
                            "logloss": round(
                                float(log_loss(yv, pv)),
                                6,
                            ),
                        }
                    )

    pd.DataFrame(model_rows).to_csv(
        RESULTS / "smc_vs_dsa_model.csv", index=False
    )
    print("smc_vs_dsa_model", len(model_rows), flush=True)

    # ---------------- Phase F / K: Quantile ----------------
    qrows = []
    hi = arr["high_atr"]
    lo = arr["low_atr"]
    ct = contig
    pct = candidates[
        "quant_width_percentile_train"
    ].to_numpy(float)

    range_atr = {}
    for h in HORIZONS:
        ok = (
            np.isfinite(hi[:, :h]).all(axis=1)
            & np.isfinite(lo[:, :h]).all(axis=1)
            & ct[:, :h].all(axis=1)
        )
        r = np.full(n, np.nan)
        r[ok] = (
            np.max(hi[ok][:, :h], axis=1)
            - np.min(lo[ok][:, :h], axis=1)
        )
        range_atr[h] = r

    for s in SYMBOLS:
        sm = (candidates["symbol"] == s).to_numpy()
        rho = spearman(pct[sm], range_atr[12][sm])
        qrows.append(
            {
                "symbol": s,
                "quant_bin": "ALL",
                "n": int(sm.sum()),
                "n_with_state": int(
                    np.isfinite(pct[sm]).sum()
                ),
                "coverage": round(
                    float(np.isfinite(pct[sm]).mean()), 6
                ),
                "spearman_width_range_atr_h12": (
                    round(rho, 6)
                    if np.isfinite(rho)
                    else np.nan
                ),
                "mean_range_atr_h6": np.nan,
                "mean_range_atr_h12": np.nan,
                "mean_range_atr_h24": np.nan,
            }
        )
        for b in ("LOW", "MID", "HIGH"):
            sel = sm & (candidates["quant_bin"] == b).to_numpy()
            if sel.sum() == 0:
                continue
            ww = w[sel]
            qrows.append(
                {
                    "symbol": s,
                    "quant_bin": b,
                    "n": int(sel.sum()),
                    "n_with_state": int(sel.sum()),
                    "coverage": 1.0,
                    "spearman_width_range_atr_h12": (
                        round(rho, 6)
                        if np.isfinite(rho)
                        else np.nan
                    ),
                    "mean_range_atr_h6": round(
                        weighted_mean(range_atr[6][sel], ww), 6
                    ),
                    "mean_range_atr_h12": round(
                        weighted_mean(
                            range_atr[12][sel], ww
                        ),
                        6,
                    ),
                    "mean_range_atr_h24": round(
                        weighted_mean(
                            range_atr[24][sel], ww
                        ),
                        6,
                    ),
                }
            )
    pd.DataFrame(qrows).to_csv(
        RESULTS / "quantile_opportunity.csv", index=False
    )
    print("quantile_opportunity", len(qrows), flush=True)

    # ---------------- ambiguity ----------------
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
                    amb_rows.append(
                        {
                            "symbol": "ALL",
                            "direction": dname,
                            "stop_atr": stop,
                            "target_r": tr,
                            "horizon": h,
                            "n": int(analyzed.sum()),
                            "both_hit_rate": round(
                                float(
                                    (
                                        code == EXIT_BOTH
                                    )[analyzed].mean()
                                ),
                                6,
                            ),
                            "gap_stop_rate": round(
                                float(
                                    (
                                        code == EXIT_GAP_STOP
                                    )[analyzed].mean()
                                ),
                                6,
                            ),
                            "gap_target_rate": round(
                                float(
                                    (
                                        code
                                        == EXIT_GAP_TARGET
                                    )[analyzed].mean()
                                ),
                                6,
                            ),
                        }
                    )
    pd.DataFrame(amb_rows).to_csv(
        RESULTS / "ambiguity.csv", index=False
    )
    print("ambiguity", len(amb_rows), flush=True)

    # ---------------- summary ----------------
    audit = json.loads(
        (RESULTS / "posthoc_audit.json").read_text()
    )
    amb = pd.DataFrame(amb_rows)
    both_rate = (
        float(amb["both_hit_rate"].mean())
        if len(amb)
        else np.nan
    )

    summary = {
        "schema_version": "ob_candidate_v3_phase1",
        "parent_sha": (
            "0d3f8cc89e63c42c00e045621d291d6907304a75"
        ),
        "candidates": int(n),
        "posthoc_audit": audit,
        "4h_status": "QUARANTINED",
        "4h_reason": (
            "epoch 4h buckets contain only 1-3 x 1h components; "
            "no validated 4h authority yet"
        ),
        "primary_context_tfs": list(PRIMARY_CONTEXT_TFS),
        "quantile_coverage": {
            "n_with_state": int(
                np.isfinite(pct).sum()
            ),
            "n_total": int(n),
            "rate": round(
                float(np.isfinite(pct).mean()), 6
            ),
        },
        "ordering_ambiguity": {
            "mean_both_hit_rate": (
                round(both_rate, 6)
                if np.isfinite(both_rate)
                else None
            ),
            "granularity_verdict": (
                "5m sufficient"
                if (
                    np.isfinite(both_rate)
                    and both_rate < 0.02
                )
                else (
                    "consider 1m"
                    if (
                        np.isfinite(both_rate)
                        and both_rate > 0.10
                    )
                    else "borderline"
                )
            ),
        },
        "weighting": {
            "primary": "decision_bar_weighted",
            "robustness": "event_weighted",
        },
        "files": sorted(
            p.name
            for p in RESULTS.glob("*.csv")
        ),
    }
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("PHASE1_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
