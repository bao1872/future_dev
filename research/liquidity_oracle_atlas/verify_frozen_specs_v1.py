"""Frozen-spec verifier (READ-ONLY, fail-closed)

职责边界（与 generator 严格区分）：

    run_v2_strategy_freeze_v1.py      = generator（读 HEAD，重算，**覆写** freeze 文件）
    run_pre_p1_freeze_closure_v1.py   = incremental closure（只追加治理元数据）
    verify_frozen_specs_v1.py         = verifier（**本文件**）

本文件是 **immutable freeze verifier**：它只**读取**已提交的 frozen specs，
重新计算所有 hash 并逐项比对。任何不一致立即 fail closed。

硬约束：
  - 绝不写入 / 覆写任何 freeze 文件或任何其它文件；
  - 不读取 P1；
  - 不重新生成 spec（不调用任何 generator 的 main 路径）；
  - mismatch -> FreezeVerificationError（非零退出码）。

校验项：
  1. code_files：**git blob @ source_commit** 的 sha256（强约束）
     以及 worktree sha256（确保当前代码 == 冻结代码）
  2. input_artifacts：worktree sha256
     （这些 parquet 未被 git 跟踪，git blob 不可用 —— 已在 evidence 中显式声明）
  3. region_hash / policy_hash（RR3）与 policy_hash（E1.1）：由冻结代码**重算**
  4. critical_function_hashes：由 inspect.getsource 重算
  5. benchmark_roles：必须存在且等于冻结值
  6. benchmark numbers：必须等于历史冻结值（不得被改动）
  7. root manifest hash：由 frozen specs 重算 canonical payload 并比对
  8. P1_read 必须为 False

P1 adjudication contract 应当调用本 verifier，而不是 generator。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.liquidity_oracle_atlas import run_pre_p1_freeze_closure_v1 as closure
from research.liquidity_oracle_atlas import run_v2_strategy_freeze_v1 as gen

FREEZE_DIR = REPO_ROOT / "research/analysis_results/v2_strategy_freeze"
RR3_SPEC = FREEZE_DIR / "FROZEN_V2_REASSESS_SPEC.json"
E1_SPEC = FREEZE_DIR / "FROZEN_E1_1_MARKET_SPEC.json"
MANIFEST = FREEZE_DIR / "PRE_P1_FREEZE_MANIFEST.json"

# 历史冻结值 —— 任何改动都必须让本 verifier 失败
EXPECTED_BENCHMARK = {
    "E1.1_MARKET": {"WF1": 0.010376171415, "WF2": 0.033117784921,
                    "WF3": 0.043795014084},
    "V2_RR3_STRICT_REASSESS": {"WF1": 0.034226296623, "WF2": 0.044368531007,
                               "WF3": 0.104135080035},
    "E2_RR3_STRICT_SINGLE_LIMIT": {"WF1": -0.005963576795,
                                   "WF2": -0.016045940367,
                                   "WF3": 0.058922314077},
}


class FreezeVerificationError(RuntimeError):
    """任一冻结项不一致 —— fail closed，绝不降级为警告。"""


def _fail(msg: str) -> None:
    raise FreezeVerificationError(msg)


def _read_json(p: Path) -> dict:
    if not p.exists():
        _fail(f"MISSING_FROZEN_FILE: {p.relative_to(REPO_ROOT)}")
    return json.loads(p.read_text(encoding="utf-8"))


def _check_code_files(spec: dict, label: str, evidence: list[str]) -> None:
    source_commit = spec["source_commit"]
    for rel, expected in spec["code_files"].items():
        blob = closure.git_blob_sha256(source_commit, rel)
        if blob is None:
            _fail(f"[{label}] code file 不存在于 source_commit {source_commit}: {rel}")
        if blob != expected:
            _fail(
                f"[{label}] code blob 漂移: {rel}\n"
                f"  expected={expected}\n  git blob={blob}\n"
                f"  source_commit={source_commit}"
            )
        wt = gen.sha256_file(REPO_ROOT / rel)
        if wt != expected:
            _fail(
                f"[{label}] worktree code 漂移: {rel}\n"
                f"  expected={expected}\n  worktree={wt}"
            )
    evidence.append(f"{label}: {len(spec['code_files'])} code files "
                    f"(git blob @ {source_commit[:12]} + worktree) OK")


def _check_input_artifacts(spec: dict, label: str, evidence: list[str]) -> None:
    for rel, expected in spec["input_artifacts"].items():
        path = REPO_ROOT / rel
        if not path.exists():
            _fail(f"[{label}] 输入产物缺失: {rel}")
        actual = gen.sha256_file(path)
        if actual != expected:
            _fail(
                f"[{label}] 输入产物漂移: {rel}\n"
                f"  expected={expected}\n  actual  ={actual}"
            )
    evidence.append(f"{label}: {len(spec['input_artifacts'])} input artifacts "
                    f"(worktree sha256; untracked) OK")


def verify_frozen_specs(verbose: bool = True) -> dict[str, Any]:
    """校验全部 frozen specs。任何不一致抛 FreezeVerificationError。

    本函数只读；不会创建、覆写或删除任何文件。
    """
    evidence: list[str] = []

    rr3 = _read_json(RR3_SPEC)
    e1 = _read_json(E1_SPEC)
    manifest = _read_json(MANIFEST)

    # ---- 1/2. code + input hashes ----
    _check_code_files(rr3, "RR3", evidence)
    _check_code_files(e1, "E1.1", evidence)
    _check_input_artifacts(rr3, "RR3", evidence)
    _check_input_artifacts(e1, "E1.1", evidence)

    # ---- 3/4. 由冻结代码重算 region/policy/function hashes ----
    rr3_recomputed = gen.build_spec(rr3["source_commit"])
    if rr3_recomputed["region_hash"] != rr3["region_hash"]:
        _fail(f"region_hash 不一致: frozen={rr3['region_hash']} "
              f"recomputed={rr3_recomputed['region_hash']}")
    if rr3_recomputed["policy_hash"] != rr3["policy_hash"]:
        _fail(f"RR3 policy_hash 不一致: frozen={rr3['policy_hash']} "
              f"recomputed={rr3_recomputed['policy_hash']}")
    if rr3_recomputed["critical_function_hashes"] != rr3["critical_function_hashes"]:
        _fail("RR3 critical_function_hashes 不一致（冻结代码已被改动）")
    evidence.append(f"RR3: region_hash/policy_hash/function hashes 重算一致 "
                    f"({len(rr3['critical_function_hashes'])} functions)")

    e1_recomputed = closure.build_e1_1_spec(e1["source_commit"])
    for key in ("policy_hash", "execution_hash", "reward_hash", "timing_hash"):
        if e1_recomputed[key] != e1[key]:
            _fail(f"E1.1 {key} 不一致: frozen={e1[key]} recomputed={e1_recomputed[key]}")
    if e1_recomputed["critical_function_hashes"] != e1["critical_function_hashes"]:
        _fail("E1.1 critical_function_hashes 不一致（冻结代码已被改动）")
    evidence.append(f"E1.1: policy/execution/reward/timing hashes 重算一致 "
                    f"({len(e1['critical_function_hashes'])} functions)")

    # ---- 5. benchmark_roles ----
    if rr3.get("benchmark_roles") != closure.BENCHMARK_ROLES:
        _fail(f"benchmark_roles 缺失或漂移: {rr3.get('benchmark_roles')}")
    if e1.get("role") != closure.BENCHMARK_ROLES["E1.1_MARKET"]:
        _fail(f"E1.1 role 漂移: {e1.get('role')}")
    evidence.append("benchmark_roles OK "
                    "(E1.1=BASELINE, RR3=CANDIDATE, E2=HISTORICAL_REFERENCE)")

    # ---- dual-conflict scope ----
    if rr3.get("dual_conflict_scope") != closure.DUAL_CONFLICT_SCOPE:
        _fail("dual_conflict_scope 缺失或漂移")
    evidence.append("dual_conflict_scope OK (unification_forbidden=true)")

    # ---- 6. benchmark numbers ----
    frozen = rr3.get("frozen_benchmark")
    if frozen != EXPECTED_BENCHMARK:
        _fail(f"frozen_benchmark 数字被改动:\n  frozen={frozen}\n"
              f"  expected={EXPECTED_BENCHMARK}")
    evidence.append("frozen_benchmark 数字未改动 (9 numbers)")

    # ---- 7. root manifest hash ----
    if manifest["strategy_hashes"]["RR3_STRICT_REASSESS"]["policy_hash"] != rr3["policy_hash"]:
        _fail("manifest.strategy_hashes(RR3) 与 frozen spec 不一致")
    if manifest["strategy_hashes"]["E1_1_HARDENED_MARKET"]["policy_hash"] != e1["policy_hash"]:
        _fail("manifest.strategy_hashes(E1.1) 与 frozen spec 不一致")
    if manifest["benchmark_roles"] != closure.BENCHMARK_ROLES:
        _fail("manifest.benchmark_roles 与冻结值不一致")
    if manifest["benchmark_numbers"] != {
        "V2_RR3_STRICT_REASSESS": frozen["V2_RR3_STRICT_REASSESS"],
        "E1.1_MARKET": frozen["E1.1_MARKET"],
        "E2_RR3_STRICT_SINGLE_LIMIT": frozen["E2_RR3_STRICT_SINGLE_LIMIT"],
    }:
        _fail("manifest.benchmark_numbers 与 frozen spec 不一致")
    if manifest["source_commit"] != rr3["source_commit"]:
        _fail("manifest.source_commit 与 frozen spec 不一致")

    payload = dict(
        strategy_hashes=manifest["strategy_hashes"],
        code_hashes=manifest["code_hashes"],
        input_hashes=manifest["input_hashes"],
        benchmark_roles=manifest["benchmark_roles"],
        benchmark_numbers=manifest["benchmark_numbers"],
        source_commit=manifest["source_commit"],
    )
    recomputed_manifest_hash = gen.canonical_sha(payload)
    if recomputed_manifest_hash != manifest["freeze_manifest_hash"]:
        _fail(f"freeze_manifest_hash 不一致:\n"
              f"  expected={manifest['freeze_manifest_hash']}\n"
              f"  recomputed={recomputed_manifest_hash}")
    evidence.append(f"freeze_manifest_hash 重算一致 "
                    f"({manifest['freeze_manifest_hash'][:16]}...)")

    # ---- 8. P1 未被读取 ----
    if rr3.get("P1_read") is not False or e1.get("P1_read") is not False:
        _fail("P1_read 必须为 False")
    if manifest.get("P1_read") is not False:
        _fail("manifest.P1_read 必须为 False")
    evidence.append("P1_read=false（声明层；执行层限制属 P1 contract）")

    report = dict(
        ok=True,
        source_commit=rr3["source_commit"],
        freeze_manifest_hash=manifest["freeze_manifest_hash"],
        evidence=evidence,
    )
    if verbose:
        print("=" * 70)
        print("FROZEN SPEC VERIFICATION (read-only)")
        print("=" * 70)
        for line in evidence:
            print(f"  [OK] {line}")
        print(f"\n[VERDICT] VERIFIED: {report['freeze_manifest_hash']}")
        print(f"[NOTE] 本 verifier 未写入任何文件。")
    return report


def main() -> int:
    try:
        verify_frozen_specs(verbose=True)
    except FreezeVerificationError as exc:
        print(f"\n[VERDICT] FREEZE_VERIFICATION_FAILED\n{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
