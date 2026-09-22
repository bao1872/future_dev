"""Cost-robustness L2 label kernel for M1 / STRUCT44.

Task ID: FUTURE-ENTRY-VALUE-M1-STRUCT44-COST-ROBUSTNESS-V1

Phase 1 = high-efficiency cost kernel checkpoint only.

Frozen by reviewer (APPROVE WITH REQUIRED REVISIONS):
- Prepare once per symbol: raw -> 4TF -> structure / proximity + m5 ATR capture.
- 5 kappa scenarios reuse the SAME prepared context; each only re-runs the
  lightweight 6-state Bellman DP (``solve_cost_labels_fast``). Complexity:
  ``structure stream + K * DP`` (NOT ``K * (stream + DP)``).
- ``solve_cost_labels_fast`` runs the FULL frozen 6-state Bellman (necessary:
  ``Q_t(F1, a)`` depends on other position/armed states), but persists only the
  F1 (S, F, L) counterfactual Q values. It is NOT an "F1-only Bellman".
- Candidate label ``Y_L / Y_S`` are populated ONLY on the exogenous candidate
  mask (proximity_any & training_eligible & finite atr>0); non-candidate rows
  stay NaN (never -inf) so they cannot pollute label statistics or define
  candidacy via Q.
- 15-symbol kappa=0 fast output must equal the frozen R2 Oracle exactly
  (tol 1e-12 on finite Q; -inf/-inf and +inf/+inf masks must match).
- forward-ATR validity gate: each intraday unit's suffix ATR must be valid so
  zero-filled warm-up ATR never enters any formal label's future Bellman path.

Hard constraints (do NOT relax):
- Bellman unchanged; F1 definition unchanged; proximity / unit boundaries /
  execution semantics / armed-rearm unchanged; no M2; no new features.
- This module is a label kernel only. It does NOT train or evaluate models.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.experiment_structure_interaction_entry_v1 import (
    KernelCounters,
    build_base_frame,
    _stream_from_base,
)
from research.liquidity_oracle_atlas.build_structure_constrained_trade_oracle_dp_v2 import (
    S2I,
    STATE_POS,
    DATA_END,
    compute_proximity_episode_id,
    _solve_unit_v2,
    load_oracle_artifact_v2,
)
from research.liquidity_oracle_atlas.build_structure_constrained_trade_oracle_dp_v1 import (
    build_intraday_units,
    _unit_terminal_reason,
)

TASK_ID = "FUTURE-ENTRY-VALUE-M1-STRUCT44-COST-ROBUSTNESS-V1"

# Frozen R2 Oracle artifact root (parity target). Untracked; never committed.
R2_ARTIFACT_ROOT = Path("artifacts/intraday_dp_oracle_r2_one_entry_proximity")

# Friction grid (frozen). kappa <= 0.02 = core friction range; 0.05 = stress only.
KAPPAS: Tuple[float, ...] = (0.0, 0.005, 0.01, 0.02, 0.05)
CORE_FRICTION_MAX = 0.02
STRESS_KAPPA = 0.05

# Action column indices inside the 6x3 Q table (ACT = [-1, 0, +1]).
ACTION_S = 0
ACTION_F = 1
ACTION_L = 2
F1_INDEX = S2I[(0, 1)]

# Phase 1 universe (frozen).
SYMBOLS_15 = [
    "AG", "AU", "CU", "AL", "SN", "NI", "RB", "I", "SC",
    "RU", "MA", "TA", "M", "P", "CF",
]
PARITY_TOL = 1e-12


@dataclass(frozen=True)
class PreparedCostOracleV1:
    """Per-symbol structure context prepared ONCE and reused across all kappas.

    Numeric arrays are marked read-only in ``__post_init__`` so a single kappa
    can never mutate state shared by the other four kappas.
    """

    symbol: str
    n: int
    base: pd.DataFrame
    proximity_bits: np.ndarray
    proximity_any: np.ndarray
    atr5m: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    proximity_episode_id: np.ndarray

    def __post_init__(self) -> None:
        for arr in (
            self.proximity_bits,
            self.proximity_any,
            self.atr5m,
            self.starts,
            self.ends,
            self.proximity_episode_id,
        ):
            arr.setflags(write=False)


@dataclass
class CostKernelStats:
    """Experimental performance counters. Does NOT pollute canonical KernelCounters.

    These let the Phase-1 checkpoint mechanically PROVE ``prepare=1, DP=5`` per
    symbol instead of relying on executor self-report.
    """

    prepare_count: int = 0
    stream_count: int = 0
    dp_pass_count: int = 0
    dp_unit_count: int = 0
    dp_state_count: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "prepare_count": self.prepare_count,
            "stream_count": self.stream_count,
            "dp_pass_count": self.dp_pass_count,
            "dp_unit_count": self.dp_unit_count,
            "dp_state_count": self.dp_state_count,
        }


def prepare_cost_oracle(
    symbol: str,
    counters: KernelCounters,
    stats: CostKernelStats,
) -> PreparedCostOracleV1:
    """Load + stream structure/proximity/ATR once for a symbol.

    Must be called exactly once per symbol in a sweep; ``stats.prepare_count``
    and ``stats.stream_count`` are incremented so the performance gate can assert
    this.
    """
    stats.prepare_count += 1

    info = build_base_frame(symbol, counters)
    base = info["base"]
    stats.stream_count += 1

    res = _stream_from_base(
        base,
        info["form"],
        info["seg_completed"],
        counters,
        max_bars=None,
        symbol=symbol,
        capture_geom=False,
        capture_entry_mask=False,
        emit_events=False,
        mask_only=True,
        capture_proximity=True,
        capture_atr5m=True,
    )

    td = pd.to_datetime(base["trading_day"]).to_numpy()
    seg = base["segment"].to_numpy(np.int64)
    starts, ends = build_intraday_units(td, seg)
    prox_ep_id = compute_proximity_episode_id(
        np.asarray(res["proximity_any"], bool), starts, len(base)
    )

    return PreparedCostOracleV1(
        symbol=symbol,
        n=int(len(base)),
        base=base,
        proximity_bits=np.asarray(res["proximity_bits"], np.int64),
        proximity_any=np.asarray(res["proximity_any"], bool),
        atr5m=np.asarray(res["atr5m"], float),
        starts=np.asarray(starts, np.int64),
        ends=np.asarray(ends, np.int64),
        proximity_episode_id=prox_ep_id,
    )


def make_cost_points(atr5m: np.ndarray, kappa: float) -> np.ndarray:
    """c_t^(kappa) = kappa * ATR5m,t, clamped to 0 where ATR is not usable.

    NaN / non-positive ATR -> cost 0 (keeps the Bellman array legal). This does
    NOT legitimize those rows: the forward-ATR gate separately rejects any
    candidate whose future ATR is invalid.
    """
    atr = np.asarray(atr5m, float)
    out = np.zeros(len(atr), dtype=float)
    good = np.isfinite(atr) & (atr > 0.0)
    out[good] = float(kappa) * atr[good]
    return out


def suffix_valid_atr_mask(
    atr5m: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    """Per-unit reverse suffix ATR validity.

    For each intraday unit, walk ``t = e-1, e-2, ..., s`` and mark ``suffix_ok[t]``
    only if every ATR from ``t`` up to the forced-flat decision ``e-1`` is valid.
    Because the Bellman at candidate ``t`` may use ``cost[e-1]`` (the final
    forced-flat turnover), the suffix must hold down to ``e-1`` inclusive.
    """
    good = np.isfinite(atr5m) & (atr5m > 0.0)
    suffix_ok = np.zeros(len(atr5m), dtype=bool)
    for s, e in zip(starts, ends):
        s = int(s)
        e = int(e)
        if e - s < 2:
            continue
        ok = True
        for t in range(e - 1, s - 1, -1):
            ok = bool(ok and good[t])
            suffix_ok[t] = ok
    return suffix_ok


def solve_cost_labels_fast(
    ctx: PreparedCostOracleV1,
    kappa: float,
    stats: CostKernelStats,
) -> pd.DataFrame:
    """Full frozen 6-state Bellman; persist only F1 S/F/L counterfactual Q.

    All decision rows are emitted (needed for kappa=0 parity). ``Y_L`` / ``Y_S``
    are populated ONLY on the exogenous candidate mask; non-candidate rows keep
    NaN so they cannot enter label statistics or define candidacy via Q.
    """
    stats.dp_pass_count += 1

    base = ctx.base
    n = ctx.n
    opens = base["open"].to_numpy(float)[:n]
    times = base["time"].to_numpy()[:n]
    td = pd.to_datetime(base["trading_day"]).to_numpy()[:n]
    seg = base["segment"].to_numpy(np.int64)[:n]

    cost = make_cost_points(ctx.atr5m, kappa)

    q_f1 = np.full((n, 3), np.nan, dtype=float)
    best_f1 = np.full(n, 127, dtype=np.int8)
    eligible = np.zeros(n, dtype=bool)
    label_available = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    valid_decision = np.zeros(n, dtype=bool)

    for s, e in zip(ctx.starts, ctx.ends):
        s = int(s)
        e = int(e)
        if e - s < 2:
            continue
        terminal_reason = _unit_terminal_reason(e, n, seg)
        unit_eligible = terminal_reason != DATA_END

        core, length = _solve_unit_v2(opens, ctx.proximity_any, cost, s, e)
        stats.dp_unit_count += 1
        stats.dp_state_count += (e - s) * len(STATE_POS)

        q_f1[s:e, :] = core["Q"][:length, F1_INDEX, :]
        best_f1[s:e] = core["actions"][:length, F1_INDEX]
        eligible[s:e] = unit_eligible
        label_available[s:e] = times[e] + np.timedelta64(5, "m")
        valid_decision[s:e] = True

    idx = np.flatnonzero(valid_decision)

    atr = ctx.atr5m[idx]
    qS = q_f1[idx, ACTION_S]
    qF = q_f1[idx, ACTION_F]
    qL = q_f1[idx, ACTION_L]

    # Candidate membership.
    # Legality of F1->L / F1->S (hence finiteness of Q_F1_*) depends ONLY on
    # proximity_any / armed state / terminal forced-flat -- never on kappa. So
    # this mask is IDENTICAL across all kappas (satisfies "candidate independent
    # of kappa") and reproduces the frozen R2 candidate contract
    # (f1_full & atr_ok & eligible & proximity_any). A proximity_any==True row can
    # still be a terminal forced-flat decision where F1->L is illegal (-inf); such
    # rows are correctly excluded here. We do NOT use Q *value/sign*, best action,
    # or Y sign -- only the legality (finite/inf) of the F1 counterfactuals.
    q_finite = np.isfinite(qS) & np.isfinite(qF) & np.isfinite(qL)
    candidate = (
        ctx.proximity_any[idx]
        & eligible[idx]
        & np.isfinite(atr)
        & (atr > 0)
        & q_finite
    )

    yL = np.full(len(idx), np.nan, dtype=float)
    yS = np.full(len(idx), np.nan, dtype=float)
    yL[candidate] = (qL[candidate] - qF[candidate]) / atr[candidate]
    yS[candidate] = (qS[candidate] - qF[candidate]) / atr[candidate]
    # Formal labels MUST be finite on the candidate set.
    assert np.isfinite(yL[candidate]).all(), "candidate Y_L not finite"
    assert np.isfinite(yS[candidate]).all(), "candidate Y_S not finite"

    return pd.DataFrame({
        "symbol": ctx.symbol,
        "decision_bar_index": idx,
        "decision_time": pd.to_datetime(times[idx]) + pd.Timedelta(minutes=5),
        "trading_day": pd.to_datetime(td[idx]),
        "proximity_bits": ctx.proximity_bits[idx],
        "proximity_any": ctx.proximity_any[idx],
        "proximity_episode_id": ctx.proximity_episode_id[idx],
        "training_eligible": eligible[idx],
        "label_available_time": pd.to_datetime(label_available[idx]),
        "atr5m": atr,
        "kappa": float(kappa),
        "Q_F1_S": qS,
        "Q_F1_F": qF,
        "Q_F1_L": qL,
        "Y_S": yS,
        "Y_L": yL,
        "best_F1": best_f1[idx],
        "is_candidate": candidate,
    })


def compare_q_arrays(
    a: np.ndarray,
    b: np.ndarray,
    atol: float = PARITY_TOL,
) -> Tuple[bool, float, Dict[str, Any]]:
    """Compare two Q arrays distinguishing finite / +inf / -inf.

    ``-inf - (-inf)`` is NaN in plain subtraction, so we compare masks
    separately and only measure finite max abs error. Returns
    ``(ok, max_abs_error_on_finite, detail)``.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    finite_a = np.isfinite(a)
    finite_b = np.isfinite(b)
    posinf_a = np.isposinf(a)
    posinf_b = np.isposinf(b)
    neginf_a = np.isneginf(a)
    neginf_b = np.isneginf(b)

    finite_mask_mismatch = not np.array_equal(finite_a, finite_b)
    posinf_mask_mismatch = not np.array_equal(posinf_a, posinf_b)
    neginf_mask_mismatch = not np.array_equal(neginf_a, neginf_b)

    if finite_a.any():
        err = float(np.max(np.abs(a[finite_a] - b[finite_a])))
    else:
        err = 0.0

    ok = (
        (not finite_mask_mismatch)
        and (not posinf_mask_mismatch)
        and (not neginf_mask_mismatch)
        and err <= atol
    )
    detail = {
        "finite_mask_mismatch": int(finite_mask_mismatch),
        "posinf_mask_mismatch": int(posinf_mask_mismatch),
        "neginf_mask_mismatch": int(neginf_mask_mismatch),
        "max_abs_error_finite": err,
    }
    return ok, err, detail


def _datetime_match(a: np.ndarray, b: np.ndarray) -> int:
    a = np.asarray(a, "datetime64[ns]")
    b = np.asarray(b, "datetime64[ns]")
    both_nat = np.isnat(a) & np.isnat(b)
    diff = (a != b) & ~both_nat
    return int(diff.sum())


def parity_against_r2(
    symbol: str,
    stats: CostKernelStats,
    counters: KernelCounters,
    atol: float = PARITY_TOL,
    ctx: Optional[PreparedCostOracleV1] = None,
) -> Dict[str, Any]:
    """kappa=0 fast output vs frozen R2 Oracle (exact parity).

    If ``ctx`` is supplied it is reused (so a caller that already prepared the
    symbol does not pay the load twice)."""
    if ctx is None:
        ctx = prepare_cost_oracle(symbol, counters, stats)
    df = solve_cost_labels_fast(ctx, 0.0, stats)

    art = load_oracle_artifact_v2(R2_ARTIFACT_ROOT, symbol)
    if not art["ok"]:
        raise SystemExit(
            f"STOP_R2_ARTIFACT_LOAD_FAILED: {symbol} {art.get('reason')}"
        )
    ref = art["actions"].copy()
    ref["symbol"] = symbol

    merged = df.merge(
        ref[[
            "symbol", "decision_time", "decision_bar_index",
            "proximity_bits", "proximity_any", "training_eligible",
            "label_available_time", "Q_F1_S", "Q_F1_F", "Q_F1_L",
        ]],
        on=["symbol", "decision_time"],
        how="outer",
        indicator=True,
    )
    key_mismatch = int((merged["_merge"] != "both").sum())

    report: Dict[str, Any] = {
        "symbol": symbol,
        "key_mismatch": key_mismatch,
        "prepare_count": stats.prepare_count,
        "dp_pass_count": stats.dp_pass_count,
        "raw_load": counters.raw_load_count,
        "resample": counters.resample_count,
        "event_iters": counters.event_role_iteration_count,
    }
    if key_mismatch:
        report["ok"] = False
        return report

    for col in ("Q_F1_S", "Q_F1_F", "Q_F1_L"):
        ok, err, detail = compare_q_arrays(
            merged[f"{col}_x"].to_numpy(float),
            merged[f"{col}_y"].to_numpy(float),
            atol,
        )
        report[f"{col}_ok"] = ok
        report[f"{col}_detail"] = detail

    for col in ("decision_bar_index", "proximity_bits"):
        report[f"{col}_mismatch"] = int(
            (merged[f"{col}_x"].to_numpy() != merged[f"{col}_y"].to_numpy()).sum()
        )
    for col in ("proximity_any", "training_eligible"):
        report[f"{col}_mismatch"] = int(
            (
                merged[f"{col}_x"].to_numpy(bool)
                != merged[f"{col}_y"].to_numpy(bool)
            ).sum()
        )
    report["label_available_time_mismatch"] = _datetime_match(
        merged["label_available_time_x"].to_numpy(),
        merged["label_available_time_y"].to_numpy(),
    )

    report["ok"] = bool(
        key_mismatch == 0
        and all(report[f"{c}_ok"] for c in ("Q_F1_S", "Q_F1_F", "Q_F1_L"))
        and all(
            report[f"{c}_mismatch"] == 0
            for c in (
                "decision_bar_index", "proximity_bits", "proximity_any",
                "training_eligible", "label_available_time",
            )
        )
    )
    return report


def run_cost_kernel_checkpoint(
    artifact_dir: str = "artifacts/entry_value_cost_robustness_m1_v1",
) -> Dict[str, Any]:
    """Phase-1 kernel checkpoint: parity + forward-ATR gate + performance gate.

    Produces three small CSVs (no large parquet). Raises SystemExit on any
    hard STOP condition so the checkpoint cannot be silently weakened.
    """
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) 15-symbol kappa=0 exact parity against frozen R2 Oracle ----
    #        + forward-ATR gate on the frozen candidate contract.
    #        Each symbol is prepared EXACTLY ONCE and reused for both checks.
    parity_rows: List[Dict[str, Any]] = []
    for sym in SYMBOLS_15:
        counters = KernelCounters()
        stats = CostKernelStats()
        ctx = prepare_cost_oracle(sym, counters, stats)
        rep = parity_against_r2(sym, stats, counters, atol=PARITY_TOL, ctx=ctx)
        parity_rows.append(rep)
        if not rep["ok"]:
            raise SystemExit(f"STOP_COST_PARITY_FAIL: {sym}")

        # Forward-ATR gate: candidate indices come from the FROZEN R2 contract
        # (proximity_any & training_eligible), joined to current causal ATR.
        art = load_oracle_artifact_v2(R2_ARTIFACT_ROOT, sym)
        r2 = art["actions"]
        r2_cand = r2[
            r2["proximity_any"].fillna(False).to_numpy(bool)
            & r2["training_eligible"].fillna(False).to_numpy(bool)
        ]
        cand_bar = r2_cand["decision_bar_index"].to_numpy(int)
        suffix_ok = suffix_valid_atr_mask(ctx.atr5m, ctx.starts, ctx.ends)
        if not np.all(suffix_ok[cand_bar]):
            raise SystemExit("STOP_COST_FORWARD_ATR_INVALID")
    parity_df = pd.DataFrame(parity_rows)
    parity_df.to_csv(out_dir / "cost_kernel_parity.csv", index=False)

    # ---- 3) AU / RB 5-kappa performance gate (prepare once, DP x5) ----
    runtime_rows: List[Dict[str, Any]] = []
    coverage_rows: List[Dict[str, Any]] = []
    for sym in ("AU", "RB"):
        counters = KernelCounters()
        stats = CostKernelStats()
        t0 = time.perf_counter()
        ctx = prepare_cost_oracle(sym, counters, stats)
        t_prep = time.perf_counter() - t0

        for kappa in KAPPAS:
            tc = time.perf_counter()
            df = solve_cost_labels_fast(ctx, kappa, stats)
            dt = time.perf_counter() - tc
            cand = df["is_candidate"].to_numpy(bool)
            n_dec = int(len(df))
            n_cand = int(cand.sum())
            coverage_rows.append({
                "symbol": sym,
                "kappa": kappa,
                "n_decisions": n_dec,
                "n_candidate": n_cand,
                "candidate_rate": float(cand.mean()) if n_dec else 0.0,
                "y_finite_rate": (
                    float(np.isfinite(df["Y_L"].to_numpy()).mean())
                    if n_dec else 0.0
                ),
            })
            runtime_rows.append({
                "symbol": sym,
                "kappa": kappa,
                "prepare_sec": t_prep,
                "dp_sec": dt,
                "n_decisions": n_dec,
                "n_candidate": n_cand,
                "raw_load": counters.raw_load_count,
                "resample": counters.resample_count,
                "event_iters": counters.event_role_iteration_count,
                "dp_unit_count": stats.dp_unit_count,
                "dp_state_count": stats.dp_state_count,
                "prepare_count": stats.prepare_count,
                "dp_pass_count": stats.dp_pass_count,
            })

        # Performance gate: mechanically PROVE prepare=1, DP=5, no event engine.
        assert stats.prepare_count == 1, f"{sym} prepare_count={stats.prepare_count}"
        assert stats.stream_count == 1, f"{sym} stream_count={stats.stream_count}"
        assert stats.dp_pass_count == 5, f"{sym} dp_pass_count={stats.dp_pass_count}"
        assert counters.raw_load_count == 1, f"{sym} raw_load={counters.raw_load_count}"
        assert counters.resample_count == 4, f"{sym} resample={counters.resample_count}"
        assert counters.full_history_recompute_count == 0
        assert counters.event_role_iteration_count == 0
        assert counters.event_classifier_call_count == 0
        assert counters.outcome_call_count == 0

    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(out_dir / "cost_coverage.csv", index=False)
    runtime_df = pd.DataFrame(runtime_rows)
    runtime_df.to_csv(out_dir / "cost_runtime.csv", index=False)

    summary = {
        "task_id": TASK_ID,
        "parity_symbols": len(parity_rows),
        "parity_all_ok": bool(parity_df["ok"].all()),
        "forward_atr_gate": "PASS",
        "performance_gate": "PASS",
    }
    (out_dir / "cost_kernel_checkpoint_summary.json").write_text(
        __import__("json").dumps(summary, indent=2)
    )
    return summary


if __name__ == "__main__":
    print(run_cost_kernel_checkpoint())
