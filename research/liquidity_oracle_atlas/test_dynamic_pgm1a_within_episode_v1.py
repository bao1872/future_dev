"""Regression test: DYNAMIC-PGM-1A must stay byte-identical to its frozen commit.

1A (Gaussian-count, 9-dim continuous) is FROZEN at 20fbd2e. The support-correct
Hurdle-Poisson variant lives in 1A.1. This guard prevents silently folding 1A.1
back into 1A, which would break reproducibility of the existing dynamic_pgm1a_*
results produced from the frozen code.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

FROZEN_SHA = "20fbd2e487dbe5cc9ce22ec1205cc7a84227b8fb"
REL = "research/liquidity_oracle_atlas/experiment_dynamic_pgm1a_within_episode_v1.py"


def test_1a_byte_identical_to_frozen_commit():
    frozen = subprocess.check_output(["git", "show", f"{FROZEN_SHA}:{REL}"])
    actual = (REPO / REL).read_bytes()
    assert frozen == actual, "1A drifted from frozen 20fbd2e commit"


def test_1a_is_gaussian_count_9dim():
    import research.liquidity_oracle_atlas.experiment_dynamic_pgm1a_within_episode_v1 as m
    assert len(m.CONT_Z) == 9, "frozen 1A must keep 9-dim Gaussian CONT_Z"
    assert not hasattr(m, "ZeroTruncatedPoissonRegressor"), \
        "1A must NOT carry the 1A.1 ZTP machinery"
    assert not hasattr(m, "COUNT_Z"), "1A must not define a separate COUNT_Z block"


if __name__ == "__main__":
    test_1a_byte_identical_to_frozen_commit()
    test_1a_is_gaussian_count_9dim()
    print("test_dynamic_pgm1a_within_episode_v1: ALL OK")
