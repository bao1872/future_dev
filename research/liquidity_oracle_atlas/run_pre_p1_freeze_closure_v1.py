"""Pre-P1 Freeze Closure  (governance, NOT optimization)

本脚本在 P1 之前补齐治理缺口。它**不**优化、**不**调参、**不**新增策略自由度、
**不**读取 P1。它只做五件事：

  1) 为 **E1.1 Hardened Market** 建立与 RR3 对称的 freeze spec
     （execution / reward / timing / P1 generation contract / function & code & input hashes
      + executable development replay）。
  2) 给 RR3 freeze **增加 `benchmark_roles`**（数字一个不改）：
        E1.1_MARKET                -> P1_ADJUDICATION_BASELINE
        V2_RR3_STRICT_REASSESS     -> P1_ADJUDICATION_CANDIDATE
        E2_RR3_STRICT_SINGLE_LIMIT -> HISTORICAL_SEMANTIC_REFERENCE
  3) 明确 **dual-conflict scope**（build_policy != route_reassess；禁止统一）。
  4) 计算 **root manifest hash**（freeze_manifest_hash）。
  5) 写出 closure 包。

与 `run_v2_strategy_freeze_v1.py` 的分工：
  那个脚本是 **generator**（读 HEAD 重算并覆写 RR3 spec）。
  本脚本只做 **增量补全**：RR3 的数字 / region_hash / policy_hash 一律不改，
  仅追加治理元数据；并新增 E1.1 对称 spec 与 root manifest。
  只读校验请使用 `verify_frozen_specs_v1.py`（不写任何文件）。

P1_read=false。本脚本不构造任何新策略，也不改变任何 reward / execution / geometry 语义。
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

from research.liquidity_oracle_atlas import run_enter_skip_selection_v1 as ess
from research.liquidity_oracle_atlas import run_liquidity_field_action_surface_v1 as s4a
from research.liquidity_oracle_atlas import run_v2_ml_recency_multi_action_v1 as v2

OUT = REPO_ROOT / "research/analysis_results/v2_strategy_freeze"

# ---------------------------------------------------------------------------
# 冻结源提交（与 RR3 spec 同一点，保证 code blob 校验可复算）
# 注意：不取 `git rev-parse HEAD` —— 本脚本是增量补全，不是 generator。
# 未来是否会漂移由 verify_frozen_specs_v1.py 的 code-blob 校验负责发现。
# ---------------------------------------------------------------------------
SOURCE_COMMIT = "7691ff21f3d9552151cf3df3ae48e2420aa3ae92"

PARITY_ATOL = 1e-9
BASELINE_ATOL = 1e-4

# ---- E1.1 Hardened Market（lag-1 market execution）----
EXECUTION_LAG_BARS = 1
OUTCOME_WINDOW_BARS = 34

E1_1_CODE_FILES = [
    "research/liquidity_oracle_atlas/run_execution_frontier_v1.py",
    "research/liquidity_oracle_atlas/run_fixed_execution_baseline_v1.py",
    "research/liquidity_oracle_atlas/run_liquidity_field_action_surface_v1.py",
    "research/liquidity_oracle_atlas/run_enter_skip_selection_v1.py",
    "research/liquidity_oracle_atlas/run_v2_ml_recency_multi_action_v1.py",
]
E1_1_CRITICAL_FUNCTIONS = [
    ("conservative_reward", v2.conservative_reward),
    ("first_hit_bounds", s4a.first_hit_bounds),
    ("surviving_field_and_target", s4a.surviving_field_and_target),
    ("latest_confirmed_extreme", s4a.latest_confirmed_extreme),
    ("build_policy", ess.build_policy),
    ("primary_matches", ess.primary_matches),
]
E1_1_INPUT_ARTIFACTS = [
    "research/analysis_results/smc_oracle_atlas_v1/liquidity_master_v1_1.parquet",
    "research/analysis_results/smc_oracle_atlas_v1/liquidity_contacts_v1_1.parquet",
    "research/analysis_results/execution_frontier_v1/execution_lag1_trades.parquet",
]

BENCHMARK_ROLES = {
    "E1.1_MARKET": "P1_ADJUDICATION_BASELINE",
    "V2_RR3_STRICT_REASSESS": "P1_ADJUDICATION_CANDIDATE",
    "E2_RR3_STRICT_SINGLE_LIMIT": "HISTORICAL_SEMANTIC_REFERENCE",
}

DUAL_CONFLICT_SCOPE = {
    "build_policy": (
        "TB1 / development opportunity-policy construction semantics. "
        "R2+R3 both matched -> SKIP_DUAL_ACTION_CONFLICT and the elif-chain ends, "
        "so R4 is NOT reachable in that branch."
    ),
    "route_reassess": (
        "RR3 Strict Reassess execution lifecycle semantics. R2+R3 both available -> "
        "neither attempt is made (transition R2_R3_DUAL_CONFLICT recorded), but R4 IS "
        "still routed afterwards (separate `if R4 in avail` block)."
    ),
    "unification_forbidden": True,
    "p1_uses": "route_reassess",
    "reason": (
        "两种语义服务不同阶段：build_policy 决定开发期机会策略构造；"
        "route_reassess 决定 P1 候选策略的执行生命周期。"
        "在 P1 前统一二者等同于修改策略，禁止。"
    ),
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical_sha(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def git_blob_sha256(commit: str, path: str) -> str | None:
    """`git cat-file -p <commit>:<path>` 的 sha256；不存在返回 None。"""
    proc = subprocess.run(
        ["git", "cat-file", "-p", f"{commit}:{path}"],
        cwd=REPO_ROOT, capture_output=True,
    )
    if proc.returncode != 0:
        return None
    return sha256_bytes(proc.stdout)


# ---------------------------------------------------------------------------
# E1.1 spec
# ---------------------------------------------------------------------------
def build_e1_1_spec(source_commit: str) -> dict:
    execution_spec = dict(
        execution="E1_1_HARDENED_MARKET",
        execution_lag_bars=EXECUTION_LAG_BARS,
        entry=(
            "membership known at the next open -> executable entry delayed to the "
            "following full 5m bar open"
        ),
        signal_bar_index="contact_bar_index + 1 + h",
        entry_bar_index="signal_bar_index + 1",
        actionable_entry="open of entry_bar_index",
        outcome_window_bars=OUTCOME_WINDOW_BARS,
        fill_model="MARKET_AT_OPEN (no limit price, no fill simulation)",
        gate_a_waiting_bar=(
            "during the one full waiting bar (signal_bar_index) the frozen thesis must "
            "remain alive; because no position exists yet, either boundary invalidates "
            "the signal with zero economic return, so ordering is irrelevant when both "
            "are touched"
        ),
        prefill_statuses=[
            "BOTH_BOUNDARIES_TOUCHED_BEFORE_ENTRY",
            "TARGET_CONSUMED_BEFORE_ENTRY",
            "STOP_INVALIDATED_BEFORE_ENTRY",
            "DISCONTINUITY_BEFORE_ENTRY",
        ],
        geometry_invalid_status="LAG1_GAP_ENTRY_INVALID",
        executed_status="EXECUTED_LAG1",
        discontinuity_rule=(
            "within the 34-bar outcome window, bars after the first discontinuity are "
            "nulled before first_hit_bounds"
        ),
        whole_policy_gate=(
            "verdict=LAG1_HARDENED_EXECUTION_EDGE_SURVIVES requires "
            "EV_R_lower_per_signal > 0 AND EV_R_lower_censor_worst_per_signal > 0 "
            "in ALL three development WFs"
        ),
    )

    reward_spec = dict(
        reward_fn="conservative_reward(filled, R_lower, censored)",
        filled="trades.status == 'EXECUTED_LAG1'",
        resolved="filled & ~censored -> R_lower",
        filled_and_censored="-1.0",
        not_filled="0.0",
        note=(
            "conservative_reward 仅对 non-fill 携带 R_lower == 0 的 action 有效；"
            "不适用于 frozen E2 SINGLE_ATTEMPT limit（后者用 "
            "e2_single_limit_censor_worst）。"
        ),
    )

    timing_spec = dict(
        t0="contact decision_time (liquidity field frozen at t0)",
        signal_bar_index="contact_bar_index + 1 + h",
        validation_bar="signal_bar_index (E1.1 Gate A waiting bar)",
        entry_bar_index="signal_bar_index + 1",
        actionable_entry="open of entry_bar_index",
        reward_end_bar_index="entry_bar_index + 33",
        reward_end_time=(
            "authoritative BAR END of reward_end_bar_index "
            "(= bar_start_time + 5min; availability_time == bar_end_time)"
        ),
        purge="train reward_end_time < validation_start",
        bar_label="INTERVAL END; bar_start_time = bar_end - 5min",
    )

    p1_generation_contract = dict(
        purpose=(
            "拿到全新 P1 数据后，E1.1 必须按以下**不可变**规则生成结果；"
            "不允许任何搜索、优化或选择性重跑。"
        ),
        step_1_policy=(
            "build_policy(surface, primary_matches(surface), meta) —— "
            "R1–R4 geometry 与 precedence 冻结不变"
        ),
        step_2_membership="R1–R4 membership frozen and next-open-conditioned",
        step_3_execution="entry delayed to the following full 5m bar open (execution_lag_bars=1)",
        step_4_prefill_gate="E1.1 Gate A on the single waiting bar",
        step_5_outcome="first_hit_bounds over 34 bars from entry bar; discontinuity truncation",
        step_6_reward="conservative_reward(filled, R_lower, censored)",
        step_7_metric="per-WF mean over ALL signals in that WF (not only executed)",
        forbidden=[
            "any optimizer / parameter search",
            "oracle or ML action selection",
            "S2 / time-decay / best-symbol / region diagnostics",
            "reading P1 before the preregistered adjudication contract is committed",
        ],
    )

    func_hashes = {
        n: sha256_bytes(inspect.getsource(f).encode())
        for n, f in E1_1_CRITICAL_FUNCTIONS
    }
    execution_hash = canonical_sha(execution_spec)
    reward_hash = canonical_sha(reward_spec)
    timing_hash = canonical_sha(timing_spec)
    policy_hash = canonical_sha(dict(
        execution_hash=execution_hash,
        reward_hash=reward_hash,
        timing_hash=timing_hash,
        critical_function_hashes=func_hashes,
    ))

    code_files = {f: sha256_file(REPO_ROOT / f) for f in E1_1_CODE_FILES}
    input_artifacts = {p: sha256_file(REPO_ROOT / p) for p in E1_1_INPUT_ARTIFACTS}

    return dict(
        candidate="E1_1_HARDENED_MARKET",
        role=BENCHMARK_ROLES["E1.1_MARKET"],
        status="DEVELOPMENT_FROZEN_PENDING_P1",
        naming_constraint=(
            "development baseline only; NOT 'optimal', 'final', 'validated' or "
            "'live' strategy"
        ),
        source_commit=source_commit,
        frozen_at=dict(
            relative_to_commit=source_commit,
            note=(
                "E1.1 的语义在源提交处被记录并锁定；本 spec 由 pre-P1 closure 生成，"
                "不读取 HEAD，也不覆盖 RR3 spec。"
            ),
        ),
        region_hash=None,
        region_hash_absent_reason=(
            "E1.1 是 market-execution baseline，不存在 R1–R4 opportunity-policy regions，"
            "因此没有 region_hash；其身份由 execution/reward/timing + function hashes "
            "构成的 policy_hash 定义。"
        ),
        market_execution_policy=execution_spec,
        reward_outcome_semantics=reward_spec,
        timing_contract=timing_spec,
        p1_generation_contract=p1_generation_contract,
        execution_hash=execution_hash,
        reward_hash=reward_hash,
        timing_hash=timing_hash,
        policy_hash=policy_hash,
        critical_function_hashes=func_hashes,
        code_files=code_files,
        input_artifacts=input_artifacts,
        input_artifact_provenance=(
            "worktree sha256 only —— 这三个 parquet 未被 git 跟踪，"
            "因此无法用 git blob 校验；hash 是唯一可用证据。"
        ),
        frozen_benchmark={},
        benchmark_tolerance=PARITY_ATOL,
        P1_read=False,
    )


# ---------------------------------------------------------------------------
# E1.1 executable development replay
# ---------------------------------------------------------------------------
def run_e1_1_replay(baseline: dict) -> dict:
    """从冻结输入产物 + 冻结 reward 函数重放 E1.1 开发基准。

    只做 E1.1 需要的 Market baseline 重放（与 run_stage_b0 的 Market 段同构）：
      filled   = status == 'EXECUTED_LAG1'
      censored = trades.censored
      reward   = conservative_reward(filled, R_lower, censored)
      EV_cw    = reward.mean()                (per WF, 全部信号)
      EV_lower = sum(R_lower of executed) / n (per WF, 全部信号)
      total_R  = sum(R_lower of executed)     (per WF)
    不重建 ML dataset —— E1.1 不需要它。
    """
    trades = pd.read_parquet(
        REPO_ROOT / "research/analysis_results/execution_frontier_v1/"
                    "execution_lag1_trades.parquet"
    )
    m_filled = (trades["status"] == "EXECUTED_LAG1").to_numpy()
    m_censored = trades["censored"].fillna(False).astype(bool).to_numpy()
    m_rlow = trades["R_lower"].fillna(0.0).to_numpy()
    cw = v2.conservative_reward(m_filled, m_rlow, m_censored)

    out: dict[str, dict] = {}
    ok = True
    for wf in ("WF1", "WF2", "WF3"):
        sub = trades[trades["wf"] == wf]
        idx = sub.index
        n = len(sub)
        executed = sub["status"] == "EXECUTED_LAG1"
        ev_lower = float(sub.loc[executed, "R_lower"].sum() / n)
        ev_cw = float(cw[idx].mean())
        total_r = float(sub.loc[executed, "R_lower"].sum())

        e1_frozen = baseline["frozen_benchmark"]["E1.1_MARKET"][wf]
        ref = baseline["baseline_e1_market"][wf]

        diff_frozen = abs(ev_cw - e1_frozen)
        diff_baseline = max(
            abs(ev_lower - ref["EV_lower"]),
            abs(ev_cw - ref["EV_cw"]),
            abs(total_r - ref["total_R"]),
        )
        good = (diff_frozen < PARITY_ATOL) and (diff_baseline < BASELINE_ATOL)
        ok &= good
        out[wf] = dict(
            n_signals=int(n),
            EV_cw=dict(actual=ev_cw, expected=e1_frozen, abs_diff=diff_frozen,
                       pass_=bool(diff_frozen < PARITY_ATOL)),
            EV_lower=dict(actual=ev_lower, expected=ref["EV_lower"],
                          abs_diff=abs(ev_lower - ref["EV_lower"]),
                          pass_=bool(abs(ev_lower - ref["EV_lower"]) < BASELINE_ATOL)),
            total_R=dict(actual=total_r, expected=ref["total_R"],
                         abs_diff=abs(total_r - ref["total_R"]),
                         pass_=bool(abs(total_r - ref["total_R"]) < BASELINE_ATOL)),
            pass_=bool(good),
        )
    return dict(replay_ok=bool(ok), n_signals_total=int(len(trades)), by_wf=out)


# ---------------------------------------------------------------------------
# RR3 spec / audit patch（仅追加治理元数据；数字一个不改）
# ---------------------------------------------------------------------------
def patch_rr3_spec(spec: dict) -> dict:
    if spec.get("frozen_benchmark") != {
        "E1.1_MARKET": {"WF1": 0.010376171415, "WF2": 0.033117784921, "WF3": 0.043795014084},
        "V2_RR3_STRICT_REASSESS": {"WF1": 0.034226296623, "WF2": 0.044368531007,
                                   "WF3": 0.104135080035},
        "E2_RR3_STRICT_SINGLE_LIMIT": {"WF1": -0.005963576795, "WF2": -0.016045940367,
                                       "WF3": 0.058922314077},
    }:
        raise SystemExit("STOP_PRE_P1_CLOSURE_RR3_BENCHMARK_UNEXPECTED")

    new: dict = {}
    for k, val in spec.items():
        new[k] = val
        if k == "execution_policy":
            new["dual_conflict_scope"] = DUAL_CONFLICT_SCOPE
        if k == "frozen_benchmark":
            new["benchmark_roles"] = BENCHMARK_ROLES
    if "benchmark_roles" not in new:
        new["benchmark_roles"] = BENCHMARK_ROLES
    if "dual_conflict_scope" not in new:
        new["dual_conflict_scope"] = DUAL_CONFLICT_SCOPE
    return new


def patch_rr3_audit(audit: dict) -> dict:
    new: dict = {}
    for k, val in audit.items():
        new[k] = val
        if k == "frozen_benchmark":
            new["benchmark_roles"] = BENCHMARK_ROLES
    if "benchmark_roles" not in new:
        new["benchmark_roles"] = BENCHMARK_ROLES
    return new


# ---------------------------------------------------------------------------
# root manifest
# ---------------------------------------------------------------------------
def build_manifest(rr3_spec: dict, e1_spec: dict,
                   spec_hashes: dict[str, str]) -> dict:
    strategy_hashes = {
        "RR3_STRICT_REASSESS": {
            "region_hash": rr3_spec["region_hash"],
            "policy_hash": rr3_spec["policy_hash"],
        },
        "E1_1_HARDENED_MARKET": {
            "region_hash": None,
            "policy_hash": e1_spec["policy_hash"],
        },
    }
    code_hashes: dict[str, str] = {}
    input_hashes: dict[str, str] = {}
    for spec in (rr3_spec, e1_spec):
        code_hashes.update(spec["code_files"])
        input_hashes.update(spec["input_artifacts"])

    benchmark_numbers = {
        "V2_RR3_STRICT_REASSESS": rr3_spec["frozen_benchmark"]["V2_RR3_STRICT_REASSESS"],
        "E1.1_MARKET": rr3_spec["frozen_benchmark"]["E1.1_MARKET"],
        "E2_RR3_STRICT_SINGLE_LIMIT": rr3_spec["frozen_benchmark"]["E2_RR3_STRICT_SINGLE_LIMIT"],
    }
    source_commit = rr3_spec["source_commit"]
    if e1_spec["source_commit"] != source_commit:
        raise SystemExit("STOP_PRE_P1_CLOSURE_SOURCE_COMMIT_MISMATCH")

    payload = dict(
        strategy_hashes=strategy_hashes,
        code_hashes=code_hashes,
        input_hashes=input_hashes,
        benchmark_roles=BENCHMARK_ROLES,
        benchmark_numbers=benchmark_numbers,
        source_commit=source_commit,
    )
    return dict(
        manifest="PRE_P1_FREEZE_MANIFEST",
        generated_by="research/liquidity_oracle_atlas/run_pre_p1_freeze_closure_v1.py",
        verified_by="research/liquidity_oracle_atlas/verify_frozen_specs_v1.py",
        source_commit=source_commit,
        frozen_spec_files=dict(
            RR3="research/analysis_results/v2_strategy_freeze/"
                "FROZEN_V2_REASSESS_SPEC.json",
            E1_1="research/analysis_results/v2_strategy_freeze/"
                 "FROZEN_E1_1_MARKET_SPEC.json",
        ),
        frozen_spec_sha256=spec_hashes,
        strategy_hashes=strategy_hashes,
        code_hashes=code_hashes,
        input_hashes=input_hashes,
        input_artifact_provenance=(
            "worktree sha256 only (untracked parquet; no git blob available)"
        ),
        benchmark_roles=BENCHMARK_ROLES,
        benchmark_numbers=benchmark_numbers,
        freeze_manifest_hash=canonical_sha(payload),
        freeze_manifest_hash_inputs=[
            "strategy_hashes", "code_hashes", "input_hashes",
            "benchmark_roles", "benchmark_numbers", "source_commit",
        ],
        P1_read=False,
        governance=dict(
            strategy_changes_in_closure=False,
            optimization_performed=False,
            numbers_changed=False,
            p1_allowed_only_after_freeze_review=True,
            verifier_is_read_only=True,
        ),
    )


def write_rr3_md_note(rr3_spec: dict, e1_spec: dict, manifest: dict,
                      replay: dict) -> str:
    L: list[str] = []
    A = L.append
    A("# Pre-P1 Freeze Closure\n")
    A("本文件由 `run_pre_p1_freeze_closure_v1.py` 生成（governance only）。")
    A("RR3 的数字 / `region_hash` / `policy_hash` **未被改动**；此处只补 E1.1 对称 freeze、")
    A("`benchmark_roles`、dual-conflict scope 与 root manifest。\n")
    A(f"- source_commit: `{manifest['source_commit']}`")
    A(f"- `freeze_manifest_hash` = `{manifest['freeze_manifest_hash']}`\n")

    A("## 1. Benchmark roles\n")
    A("| benchmark | role |")
    A("|---|---|")
    for k, v in BENCHMARK_ROLES.items():
        A(f"| `{k}` | `{v}` |")
    A("")
    A("> P1 只允许取 `P1_ADJUDICATION_BASELINE` 与 `P1_ADJUDICATION_CANDIDATE`。")
    A("> `HISTORICAL_SEMANTIC_REFERENCE` 作为语义回归哨兵保留（例如防止")
    A("> `AMBIGUOUS_FILL_STOP_ORDER` 的 -1R 被再次抹成 0R），不得作为 P1 比较对象。\n")

    A("## 2. E1.1 Hardened Market（对称 freeze）\n")
    A(f"- `policy_hash` = `{e1_spec['policy_hash']}`")
    A(f"- `execution_hash` = `{e1_spec['execution_hash']}`")
    A(f"- `reward_hash` = `{e1_spec['reward_hash']}`")
    A(f"- `timing_hash` = `{e1_spec['timing_hash']}`")
    A(f"- `region_hash` = `None`（{e1_spec['region_hash_absent_reason']}）\n")
    A("### executable development replay\n")
    A(f"replay_ok = **{replay['replay_ok']}** (n_signals={replay['n_signals_total']})\n")
    rows = []
    for wf, d in replay["by_wf"].items():
        rows.append(dict(wf=wf, n_signals=d["n_signals"],
                         EV_cw_expected=d["EV_cw"]["expected"],
                         EV_cw_actual=d["EV_cw"]["actual"],
                         EV_cw_abs_diff=d["EV_cw"]["abs_diff"],
                         EV_lower_expected=d["EV_lower"]["expected"],
                         EV_lower_actual=d["EV_lower"]["actual"],
                         total_R_expected=d["total_R"]["expected"],
                         total_R_actual=d["total_R"]["actual"],
                         pass_=d["pass_"]))
    A("```\n" + pd.DataFrame(rows).to_string(index=False) + "\n```\n")

    A("## 3. Dual-conflict scope（原样冻结，禁止统一）\n")
    A("| 函数 | 语义 |")
    A("|---|---|")
    A(f"| `build_policy` | {DUAL_CONFLICT_SCOPE['build_policy']} |")
    A(f"| `route_reassess` | {DUAL_CONFLICT_SCOPE['route_reassess']} |")
    A("")
    A(f"- `unification_forbidden` = `{DUAL_CONFLICT_SCOPE['unification_forbidden']}`")
    A(f"- P1 使用：`{DUAL_CONFLICT_SCOPE['p1_uses']}`")
    A(f"- 理由：{DUAL_CONFLICT_SCOPE['reason']}\n")

    A("## 4. Root manifest hash\n")
    A("```")
    A(f"freeze_manifest_hash = {manifest['freeze_manifest_hash']}")
    A("inputs: " + ", ".join(manifest["freeze_manifest_hash_inputs"]))
    A("```\n")
    A("## 5. Verification\n")
    A("只读校验（不写任何文件，mismatch 即 fail closed）：\n")
    A("```bash")
    A(".venv/bin/python research/liquidity_oracle_atlas/verify_frozen_specs_v1.py")
    A("```\n")
    A("## 6. 尚未完成（不在本轮范围）\n")
    A("- P1 adjudication contract（预注册）**未编写**。")
    A("- P1 **未读取**（`P1_read=false`）。")
    A("- `P1_read=false` 仍是**声明**；把「禁止读取 P1」升级为可执行限制属于 P1 contract 的职责。")
    A("")
    return "\n".join(L)


def main() -> None:
    t0 = time.perf_counter()
    print("=" * 70)
    print("Pre-P1 Freeze Closure (governance only)")
    print("=" * 70)

    rr3_path = OUT / "FROZEN_V2_REASSESS_SPEC.json"
    audit_path = OUT / "V2_STRATEGY_FREEZE_AUDIT.json"
    rr3_spec = json.loads(rr3_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    print(f"[GIT] frozen source_commit = {SOURCE_COMMIT}")

    # ---- 1) E1.1 对称 spec ----
    e1_spec = build_e1_1_spec(SOURCE_COMMIT)
    print(f"[HASH] E1.1 policy_hash={e1_spec['policy_hash'][:16]}...")

    # ---- 2) E1.1 executable development replay ----
    baseline = dict(
        frozen_benchmark=rr3_spec["frozen_benchmark"],
        baseline_e1_market={
            "WF1": {"EV_lower": 0.042022, "EV_cw": 0.010376, "total_R": 132.788702},
            "WF2": {"EV_lower": 0.047519, "EV_cw": 0.033118, "total_R": 135.286334},
            "WF3": {"EV_lower": 0.076375, "EV_cw": 0.043795, "total_R": 229.735402},
        },
    )
    replay = run_e1_1_replay(baseline)
    print(f"[REPLAY] E1.1 ok={replay['replay_ok']} "
          f"(n={replay['n_signals_total']}, {time.perf_counter()-t0:.1f}s)")
    assert replay["replay_ok"], "STOP_E1_1_FREEZE_REPLAY_FAIL"

    e1_spec["frozen_benchmark"] = {
        "EV_cw": {wf: d["EV_cw"]["actual"] for wf, d in replay["by_wf"].items()},
        "EV_lower": {wf: d["EV_lower"]["actual"] for wf, d in replay["by_wf"].items()},
        "total_R": {wf: d["total_R"]["actual"] for wf, d in replay["by_wf"].items()},
    }

    e1_replay_doc = dict(
        replay_ok=replay["replay_ok"],
        n_signals_total=replay["n_signals_total"],
        by_wf=replay["by_wf"],
        replay_purpose=(
            "prove that re-running the E1.1 market-execution replay from the frozen "
            "input artifacts and the frozen reward function reproduces the accepted "
            "E1.1 benchmark"
        ),
        replay_scope_note=(
            "仅重放 E1.1 所需的 Market baseline（与 run_stage_b0 的 Market 段同构）；"
            "不重建 ML dataset。"
        ),
        replay_command=".venv/bin/python research/liquidity_oracle_atlas/"
                       "run_pre_p1_freeze_closure_v1.py",
        tolerance=dict(parity=PARITY_ATOL, baseline=BASELINE_ATOL),
    )

    # ---- 3) patch RR3 spec / audit ----
    rr3_spec_new = patch_rr3_spec(rr3_spec)
    audit_new = patch_rr3_audit(audit)
    print("[PATCH] RR3 spec += benchmark_roles, dual_conflict_scope (numbers unchanged)")

    rr3_text = json.dumps(rr3_spec_new, indent=2, ensure_ascii=False, default=str)
    e1_text = json.dumps(e1_spec, indent=2, ensure_ascii=False, default=str)

    # ---- 4) root manifest ----
    manifest = build_manifest(
        rr3_spec_new, e1_spec,
        spec_hashes={
            "FROZEN_V2_REASSESS_SPEC.json": sha256_bytes(rr3_text.encode()),
            "FROZEN_E1_1_MARKET_SPEC.json": sha256_bytes(e1_text.encode()),
        },
    )
    print(f"[MANIFEST] freeze_manifest_hash={manifest['freeze_manifest_hash'][:16]}...")

    # ---- 5) write ----
    rr3_path.write_text(rr3_text, encoding="utf-8")
    audit_path.write_text(
        json.dumps(audit_new, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    (OUT / "FROZEN_E1_1_MARKET_SPEC.json").write_text(e1_text, encoding="utf-8")
    (OUT / "E1_1_MARKET_FREEZE_REPLAY.json").write_text(
        json.dumps(e1_replay_doc, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    (OUT / "PRE_P1_FREEZE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    (OUT / "PRE_P1_FREEZE_CLOSURE.md").write_text(
        write_rr3_md_note(rr3_spec_new, e1_spec, manifest, replay), encoding="utf-8")

    print(f"\n[VERDICT] pre-P1 freeze closure complete: E1.1 frozen, roles assigned")
    print(f"[DONE] {time.perf_counter()-t0:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
