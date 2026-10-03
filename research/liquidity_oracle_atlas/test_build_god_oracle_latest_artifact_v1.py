"""test_build_god_oracle_latest_artifact_v1
===========================================

FIX-02A artifact-contract tests. Verifies the artifact builder's hard
validation matches Option A semantics WITHOUT materializing an artifact (the
full artifact is generated separately, only after remote sign-off).

The validation is exercised against the REAL production Oracle output for AG
(full stream). It must PASS before `build_god_oracle_latest_artifact_v1.py`
is allowed to write.
"""

import pytest

from research.liquidity_oracle_atlas.structural_god_oracle_m15_v4 import (
    run_god_oracle_v4,
)
from research.liquidity_oracle_atlas.build_god_oracle_latest_artifact_v1 import (
    DIAG_SID,
    PREEMPT_EXPECT,
    PREEMPT_SID,
    _git_sha,
    _hard_validate,
)


@pytest.fixture(scope="module")
def ag_canon():
    res = run_god_oracle_v4("AG")
    canon = [r for r in res["records"] if r["canonical_oracle_trade"]]
    canon.sort(key=lambda r: int(r["candidate_start_bar"]))
    return canon


def test_builder_validation_passes_option_a(ag_canon):
    """The builder's hard validation must PASS on current production output:
    PREEMPT sentinel present + matches; DIAG absent; generic invariants hold.
    """
    t = _hard_validate(ag_canon, _git_sha())
    assert t["structure_id"] == PREEMPT_SID
    for k, exp in PREEMPT_EXPECT.items():
        got = t[k]
        if isinstance(exp, float):
            assert abs(float(got) - exp) < 1e-6, f"{k}={got} expected {exp}"
        else:
            assert str(got) == str(exp), f"{k}={got!r} expected {exp!r}"


def test_diag_absent_from_full_stream(ag_canon):
    """Option A: the diagnosed 349/350/7700/7773 case is LOCAL-regression-only
    and must NOT appear in the full AG sequential canonical stream.
    """
    assert not any(r["structure_id"] == DIAG_SID for r in ag_canon)


def test_generic_invariants(ag_canon):
    """Generic hard invariants the builder enforces before writing."""
    assert ag_canon, "expected some canonical trades"
    assert all(r["exit_reason"] == "TARGET_TOUCH" for r in ag_canon)
    assert all(float(r["utility"]) > 0 for r in ag_canon)
    for i in range(len(ag_canon) - 1):
        assert int(ag_canon[i]["exit_fill_index"]) < int(
            ag_canon[i + 1]["best_entry_decision_index"]
        )
