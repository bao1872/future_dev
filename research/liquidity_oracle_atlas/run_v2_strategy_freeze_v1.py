"""V2 Strategy Specification Freeze  (governance, NOT optimization)

本脚本不优化、不调参、不新增策略自由度。它只做三件事：

  1) 计算 region_hash / policy_hash，以及 code files / input artifacts 的 sha256
     （全部可复算：hashes 是对代码与产物的确定性函数）。
  2) 执行 §9 executable replay：从 frozen spec 重放 development benchmark，
     必须精确复现 E1.1 Market 与 V2 RR3 Strict Reassess 的每 WF EV。
  3) 写出 freeze 包：
        FROZEN_V2_REASSESS_SPEC.json
        V2_STRATEGY_FREEZE_AUDIT.json
        V2_STRATEGY_FREEZE.md
        V2_STRATEGY_FREEZE_REPLAY.json

P1 未被读取（P1_read=false）。本脚本不构造任何新策略，也不改变 reward /
execution / geometry semantics —— 只把它们**记录并锁死**。
"""
from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.liquidity_oracle_atlas import run_execution_limit_frontier_v1 as e2
from research.liquidity_oracle_atlas import run_execution_limit_frontier_v1_closure as e2c
from research.liquidity_oracle_atlas import run_enter_skip_selection_v1 as ess
from research.liquidity_oracle_atlas import run_liquidity_field_action_surface_v1 as s4a
from research.liquidity_oracle_atlas import run_v2_ml_recency_multi_action_v1 as v2
from research.liquidity_oracle_atlas.run_fixed_execution_baseline_v1 import (
    CONTACTS, MASTER, load_env)

OUT = REPO_ROOT / "research/analysis_results/v2_strategy_freeze"
OUT.mkdir(parents=True, exist_ok=True)

RR_TARGET = 3.0
FILL_MODEL = "STRICT_TRADE_THROUGH"
EXECUTION_MODE = "REASSESS_NEXT_H"
FILL_STATUSES = sorted(e2c.FILL)
TERMINAL_STATUSES = sorted(e2c.TERMINAL)

# frozen benchmark (development replay; from committed artifacts)
FROZEN_BENCHMARK = {
    "E1.1_MARKET": {"WF1": 0.010376171415, "WF2": 0.033117784921,
                    "WF3": 0.043795014084},
    "V2_RR3_STRICT_REASSESS": {"WF1": 0.034226296623, "WF2": 0.044368531007,
                               "WF3": 0.104135080035},
    "E2_RR3_STRICT_SINGLE_LIMIT": {"WF1": -0.005963576795,
                                   "WF2": -0.016045940367,
                                   "WF3": 0.058922314077},
}
PARITY_ATOL = 1e-9
CODE_FILES = [
    "research/liquidity_oracle_atlas/run_enter_skip_selection_v1.py",
    "research/liquidity_oracle_atlas/run_execution_limit_frontier_v1.py",
    "research/liquidity_oracle_atlas/run_execution_limit_frontier_v1_closure.py",
    "research/liquidity_oracle_atlas/run_liquidity_field_action_surface_v1.py",
    "research/liquidity_oracle_atlas/run_latent_state_compression_v1.py",
    "research/liquidity_oracle_atlas/run_fixed_execution_baseline_v1.py",
    "research/liquidity_oracle_atlas/run_v2_ml_recency_multi_action_v1.py",
    "research/liquidity_oracle_atlas/build_oracle_atlas_v1_2.py",
]
INPUT_ARTIFACTS = [str(MASTER), str(CONTACTS),
                   "research/analysis_results/execution_frontier_v1/"
                   "execution_lag1_trades.parquet"]
CRITICAL_FUNCTIONS = [
    ("build_policy", ess.build_policy),
    ("primary_matches", ess.primary_matches),
    ("run_block", e2.run_block),
    ("prefill", e2c.prefill),
    ("route_reassess", e2c.route_reassess),
    ("scalar_outcome", e2c.scalar_outcome),
    ("first_hit_bounds", s4a.first_hit_bounds),
    ("surviving_field_and_target", s4a.surviving_field_and_target),
    ("latest_confirmed_extreme", s4a.latest_confirmed_extreme),
    ("attach_exact_reward_end", v2.attach_exact_reward_end),
    ("conservative_reward", v2.conservative_reward),
    ("e2_single_limit_censor_worst", v2.e2_single_limit_censor_worst),
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def build_spec(source_commit: str) -> dict:
    regions = []
    for r in ess.PRIMARY_REGIONS:
        regions.append(dict(
            region=r["region"], action=r["action"], h=int(r["h"]),
            structure_scale=float(r["scale"]),
            target_atr_lo=float(r["target_lo"]), target_atr_hi=float(r["target_hi"]),
            risk_atr_lo=float(r["risk_lo"]), risk_atr_hi=float(r["risk_hi"]),
            target_atr_interval="[lo, hi)", risk_atr_interval="[lo, hi)"))
    precedence = [
        "R1 -> ENTER_R1",
        "R2 & R3 both matched -> SKIP_DUAL_ACTION_CONFLICT (no trade)",
        "R2 -> ENTER_R2", "R3 -> ENTER_R3", "R4 -> ENTER_R4",
        "no region -> SKIP_NO_GEOMETRY_EDGE",
        "at most one trade per gid (asserted MULTIPLE_ENTRY)",
    ]
    region_spec = dict(eligibility=dict(
        available=True, structure_scale_isclose=1.2,
        target_atr_lower_inclusive=True, target_atr_upper_exclusive=True,
        risk_atr_lower_inclusive=True, risk_atr_upper_exclusive=True),
        regions=regions, precedence=precedence)

    execution_spec = dict(
        rr_target=RR_TARGET, fill_model=FILL_MODEL,
        execution_mode=EXECUTION_MODE, ttl_active_bars=2,
        limit_price="(target + rr*stop) / (1 + rr)",
        limit_formula_symmetric=True,
        fill_statuses=FILL_STATUSES, terminal_statuses=TERMINAL_STATUSES,
        route=("R1 -> (if not filled and not terminal) R2 or R3 "
               "-> R4; stop at first FILL; stop at TERMINAL"),
        dual_conflict=("if both R2 and R3 available, neither is attempted; "
                       "transition R2_R3_DUAL_CONFLICT is recorded; R4 still allowed"),
        strict_trade_through=("fill requires strict inequality "
                              "(LONG: L < limit, SHORT: H > limit)"),
        marketable_open="FILLED_AT_OPEN when bar-0 open crosses the limit "
                        "(no prefill / gap / discontinuity)",
    )

    reward_spec = dict(
        target="frozen target_price", stop="frozen stop_price",
        outcome_window_bars=34,
        censor="both bounds unseen inside window -> censored",
        ambiguous_fill_target_order=dict(upper="rr", lower=0.0),
        ambiguous_fill_stop_order=dict(upper=0.0, lower=-1.0),
        prefill_statuses=sorted(e2.PRE),
        censor_worst=("per signal: mean of R_lower where filled & resolved, "
                      "-1 for filled & censored, 0 for not filled"),
        single_limit_censor_worst=("frozen E2: R_lower kept verbatim for every "
                                   "non-NaN row (incl. AMBIGUOUS_FILL_STOP_ORDER "
                                   "= -1R); NaN only for filled & censored -> -1"),
    )

    timing_spec = dict(
        t0="contact decision_time (liquidity field frozen at t0)",
        signal_bar_index="contact_bar_index + 1 + h",
        validation_bar="signal_bar_index (prefill gate bar)",
        entry_bar_index="signal_bar_index + 1",
        actionable_entry="open of entry_bar_index",
        reward_end_bar_index=("max(entry_bar+33 MARKET, entry_bar+34 LIMIT_RR3, "
                              "contact_bar+44 REASSESS_RR3)"),
        reward_end_time=("authoritative BAR END of reward_end_bar_index "
                         "(= bar_start_time + 5min; availability_time == bar_end_time)"),
        purge="train reward_end_time < validation_start",
        bar_label="INTERVAL END; bar_start_time = bar_end - 5min",
    )

    func_hashes = {n: hashlib.sha256(inspect.getsource(f).encode()).hexdigest()
                   for n, f in CRITICAL_FUNCTIONS}
    region_hash = canonical_sha(region_spec)
    policy_hash = canonical_sha(dict(
        region_hash=region_hash, execution=execution_spec,
        reward=reward_spec, timing=timing_spec, critical_function_hashes=func_hashes))

    return dict(
        candidate="RR3_STRICT_REASSESS",
        status="DEVELOPMENT_FROZEN_PENDING_P1",
        naming_constraint=("development candidate only; NOT 'optimal', 'final', "
                           "'validated' or 'live' strategy"),
        rr_target=RR_TARGET, fill_model=FILL_MODEL,
        execution_mode=EXECUTION_MODE,
        opportunity_policy=region_spec,
        execution_policy=execution_spec,
        reward_outcome_semantics=reward_spec,
        timing_contract=timing_spec,
        region_hash=region_hash, policy_hash=policy_hash,
        critical_function_hashes=func_hashes,
        source_commit=source_commit,
        code_files={f: sha256_file(REPO_ROOT / f) for f in CODE_FILES},
        input_artifacts={p: sha256_file(REPO_ROOT / p) for p in INPUT_ARTIFACTS},
        frozen_benchmark=FROZEN_BENCHMARK,
        benchmark_tolerance=PARITY_ATOL,
        superseded_oracle_ml=[dict(name="V2 absolute multi-action Q",
                                   verdict="STOP_CURRENT_MULTI_ACTION_ML",
                                   commit="a33d4eb"),
                              dict(name="V3-A1 oracle headroom (descriptive)",
                                   verdict="ORACLE_HEADROOM_MATERIAL",
                                   commit="7691ff2"),
                              dict(name="V3-A2 relative advantage learnability",
                                   verdict="NO_LEARNABLE_RELATIVE_ADVANTAGE",
                                   commit="ce1cb26")],
        P1_read=False,
    )


def run_replay() -> dict:
    """§9: re-derive the benchmark from frozen spec; numbers must not change."""
    cache = v2.OUT_DIR / "multi_action_signals_features.parquet"
    if cache.exists():
        cache.unlink()
        print(f"  [REPLAY] cache invalidated: {cache.name}")
    D, master_by_sym, bars_by_sym = load_env()
    trades = pd.read_parquet(REPO_ROOT / "research/analysis_results/"
                             "execution_frontier_v1/execution_lag1_trades.parquet")
    df_comb, _ = v2.build_multi_action_dataset(D, master_by_sym, bars_by_sym, trades)
    df = df_comb[df_comb["wf"].isin(["WF1", "WF2", "WF3"])]
    replay = {}
    ok = True
    for key, col in [("E1.1_MARKET", "reward_MARKET"),
                     ("V2_RR3_STRICT_REASSESS", "reward_REASSESS_RR3"),
                     ("E2_RR3_STRICT_SINGLE_LIMIT", "reward_LIMIT_RR3")]:
        replay[key] = {}
        for wf, exp in FROZEN_BENCHMARK[key].items():
            act = float(df.loc[df.wf == wf, col].mean())
            good = abs(act - exp) < PARITY_ATOL
            ok &= good
            replay[key][wf] = dict(actual=act, expected=exp,
                                   abs_diff=abs(act - exp), pass_=bool(good))
    return dict(replay_ok=bool(ok), n_signals=int(len(df)), by_metric=replay,
                cache_rebuilt=True)


def md_report(spec, audit, replay) -> str:
    L = []
    A = L.append
    A("# V2 Strategy Specification Freeze\n")
    A(f"- candidate: **{spec['candidate']}** — `{spec['status']}`")
    A(f"- {spec['naming_constraint']}")
    A(f"- source_commit: `{spec['source_commit']}`")
    A(f"- `region_hash` = `{spec['region_hash']}`")
    A(f"- `policy_hash` = `{spec['policy_hash']}`")
    A(f"- P1_read = `{spec['P1_read']}`\n")

    A("## 1. Opportunity policy (R1–R4, frozen)\n")
    A("```")
    for r in spec["opportunity_policy"]["regions"]:
        A(f"{r['region']}: action={r['action']} h={r['h']} scale={r['structure_scale']} "
          f"target_atr=[{r['target_atr_lo']},{r['target_atr_hi']}) "
          f"risk_atr=[{r['risk_atr_lo']},{r['risk_atr_hi']})")
    A("```")
    A("Precedence（first-match, at most one trade per gid）:\n")
    for p in spec["opportunity_policy"]["precedence"]:
        A(f"- {p}")
    A("")

    A("## 2. Execution policy\n")
    A("```\n" + json.dumps(spec["execution_policy"], indent=2) + "\n```\n")

    A("## 3. Reward / outcome semantics\n")
    A("```\n" + json.dumps(spec["reward_outcome_semantics"], indent=2) + "\n```\n")
    A("> 特别记录（a33d4eb 修复）：`AMBIGUOUS_FILL_STOP_ORDER → lower = -1R`，"
      "且**不得**被 `np.where(filled, R_lower, 0)` 抹成 0。\n")

    A("## 4. Timing contract\n")
    A("```\n" + json.dumps(spec["timing_contract"], indent=2) + "\n```\n")
    A("> 最易犯错处：某 geometry 在 next open 才可知，不可假设同一 open 成交。"
      "frozen `entry_bar_index = signal_bar_index + 1`，成交价 = 该 bar 的 open。\n")

    A("## 5. Hashes\n")
    A("### code files (sha256)\n```")
    for k, v in spec["code_files"].items():
        A(f"{v}  {k}")
    A("```")
    A("### input artifacts (sha256)\n```")
    for k, v in spec["input_artifacts"].items():
        A(f"{v}  {k}")
    A("```")
    A("### critical function hashes\n```")
    for k, v in spec["critical_function_hashes"].items():
        A(f"{v}  {k}")
    A("```\n")

    A("## 6. Frozen benchmark + §9 executable replay\n")
    A(f"replay_ok = **{replay['replay_ok']}** (n_signals={replay['n_signals']}, "
      f"cache rebuilt from frozen pipeline)\n")
    rows = []
    for key, per in replay["by_metric"].items():
        for wf, d in per.items():
            rows.append(dict(metric=key, wf=wf, expected=d["expected"],
                             actual=d["actual"], abs_diff=d["abs_diff"],
                             pass_=d["pass_"]))
    A("```\n" + pd.DataFrame(rows).to_string(index=False) + "\n```\n")

    A("## 7. Superseded / closed lines\n")
    for s in spec["superseded_oracle_ml"]:
        A(f"- {s['name']} → `{s['verdict']}` ({s['commit']})")
    A("")
    A("## 8. Freeze rules\n")
    A("- 本文件之后**不得**再改动策略自由度（geometry / execution / reward / timing）。")
    A("- 唯一允许的后续动作：P1 adjudication contract（预注册），然后一次性开封 P1。")
    A("- P1 只比较两个**预先冻结**对象：`E1.1 Hardened Market` vs "
      "`RR3 Strict Reassess`；不得带入 Oracle / ML / S2 / time-decay / "
      "best-symbol / region diagnostics。")
    A("- 本 spec 的 `status` 在 P1 完成前不得升级，名称不得脱离 "
      "**development candidate**。")
    A("")
    return "\n".join(L)


def main():
    t0 = time.perf_counter()
    print("=" * 70)
    print("V2 Strategy Specification Freeze (governance only)")
    print("=" * 70)
    src = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                         capture_output=True, text=True).stdout.strip()
    print(f"[GIT] source_commit = {src}")

    spec = build_spec(src)
    print(f"[HASH] region_hash={spec['region_hash'][:16]}... "
          f"policy_hash={spec['policy_hash'][:16]}...")
    print("[REPLAY] re-deriving benchmark from frozen pipeline ...")
    replay = run_replay()
    print(f"[REPLAY] ok={replay['replay_ok']} ({time.perf_counter()-t0:.1f}s)")
    assert replay["replay_ok"], "STOP_V2_FREEZE_REPLAY_FAIL"

    audit = dict(
        freeze="V2 Strategy Specification Freeze",
        candidate=spec["candidate"], status=spec["status"],
        source_commit=src,
        region_hash=spec["region_hash"], policy_hash=spec["policy_hash"],
        replay=replay,
        replay_ok=replay["replay_ok"],
        replay_purpose=("prove that re-running the development replay from the "
                        "frozen spec reproduces the accepted benchmark"),
        replay_command=".venv/bin/python research/liquidity_oracle_atlas/"
                       "run_v2_strategy_freeze_v1.py",
        governance=dict(P1_read=False, strategy_changes_in_freeze=False,
                        optimization_performed=False,
                        p1_allowed_only_after_freeze_review=True),
        frozen_benchmark=FROZEN_BENCHMARK,
        next_step="P1 adjudication contract (preregistered), then one-shot P1",
    )
    (OUT / "FROZEN_V2_REASSESS_SPEC.json").write_text(
        json.dumps(spec, indent=2, ensure_ascii=False, default=str))
    (OUT / "V2_STRATEGY_FREEZE_AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, default=str))
    (OUT / "V2_STRATEGY_FREEZE_REPLAY.json").write_text(
        json.dumps(replay, indent=2, ensure_ascii=False, default=str))
    (OUT / "V2_STRATEGY_FREEZE.md").write_text(md_report(spec, audit, replay),
                                              encoding="utf-8")
    print(f"\n[VERDICT] V2 candidate frozen: {spec['candidate']} "
          f"({spec['status']})")
    print(f"[DONE] {time.perf_counter()-t0:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
