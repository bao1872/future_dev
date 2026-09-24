"""FUTURE-R11-R14 V2 — feature contracts (plan §10, §16, §19, §21-§23).

TRADING_METRICS: NOT_APPLICABLE
reason: No trading action has been defined.
This module only declares column contracts and SHAs.

Everything here is declarative and frozen:

  WIN33 / PAY8 are IMPORTED from V1 (never redefined) so that A0 is by
  construction the V1 feature matrices (plan §47 #7).

  SHARED41 = WIN33 ∪ PAY8, exactly 41 columns, in that order. Nothing is
  removed for correlation reasons and no importance filtering happens (§10).

  SPACE18 / PATH8 / VOL6 are the pre-registered V2 families. Their names and
  ORDER are fixed; the builders must emit exactly these sequences.

  Variant feature counts are frozen at 41 / 59 / 49 / 47 / 73 (§23).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

# V1 ownership: imported read-only, never redefined here.
from research.liquidity_oracle_atlas.build_decomposed_value_dataset_v1 import (
    WIN33_COLS,
    PAY8_COLS,
)

# --------------------------------------------------------------------------- #
# §10 SHARED41                                                                 #
# --------------------------------------------------------------------------- #
SHARED41: tuple[str, ...] = tuple(WIN33_COLS) + tuple(PAY8_COLS)

# --------------------------------------------------------------------------- #
# §16 SPACE18 — multi-level structural space (order matches extract_space18)    #
# --------------------------------------------------------------------------- #
SPACE18: tuple[str, ...] = (
    "ahead_sr2_dist_atr",
    "ahead_sr3_dist_atr",
    "back_sr2_dist_atr",
    "back_sr3_dist_atr",

    "ahead_sr2_strength",
    "ahead_sr3_strength",
    "back_sr2_strength",
    "back_sr3_strength",

    "ahead_sr_count_1atr",
    "ahead_sr_count_2atr",
    "back_sr_count_1atr",
    "back_sr_count_2atr",

    "ahead_sr_strength_sum_2atr",
    "back_sr_strength_sum_2atr",

    "ahead_liq_dist_atr",
    "back_liq_dist_atr",
    "ahead_liq_width_atr",
    "back_liq_width_atr",
)

# --------------------------------------------------------------------------- #
# §19 PATH8 — recent path / zone interaction                                   #
# --------------------------------------------------------------------------- #
PATH8: tuple[str, ...] = (
    "ahead_touch_count_16",
    "back_touch_count_16",
    "ahead_touch_count_64",
    "back_touch_count_64",
    "log1p_bars_since_ahead_touch",
    "log1p_bars_since_back_touch",
    "candidate_count_32",
    "candidate_count_128",
)

# --------------------------------------------------------------------------- #
# §21 VOL6 — volatility regime                                                 #
# --------------------------------------------------------------------------- #
VOL6: tuple[str, ...] = (
    "atr_rel_med16",
    "atr_rel_med64",
    "rv16",
    "rv64",
    "rv_ratio_16_64",
    "range_over_atr",
)

assert len(SHARED41) == 41, len(SHARED41)
assert len(SPACE18) == 18, len(SPACE18)
assert len(PATH8) == 8, len(PATH8)
assert len(VOL6) == 6, len(VOL6)
# SHARED41 must be exactly the union of the two V1 contracts, with no overlap.
assert set(SHARED41) == set(WIN33_COLS) | set(PAY8_COLS)
assert len(set(WIN33_COLS) & set(PAY8_COLS)) == 0


# --------------------------------------------------------------------------- #
# Schema identity                                                              #
# --------------------------------------------------------------------------- #
def schema_sha256(columns) -> str:
    """SHA over the ORDERED column tuple; order changes the identity."""
    h = hashlib.sha256()
    for c in columns:
        h.update(str(c).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


@dataclass(frozen=True)
class ArchSpec:
    """A candidate architecture: what the Win head and Payoff heads consume.

    `win` is the column tuple for the classifier head; `payoff` is shared by
    BOTH magnitude regressors. For the final Composer to be coherent the two
    must coincide (plan §25).
    """

    name: str
    win: tuple[str, ...]
    payoff: tuple[str, ...]

    @property
    def n_win(self) -> int:
        return len(self.win)

    @property
    def n_payoff(self) -> int:
        return len(self.payoff)

    @property
    def shared_state(self) -> bool:
        """True when p, mu_W and mu_L are functions of the SAME causal state."""
        return tuple(self.win) == tuple(self.payoff)

    @property
    def win_schema_sha256(self) -> str:
        return schema_sha256(self.win)

    @property
    def payoff_schema_sha256(self) -> str:
        return schema_sha256(self.payoff)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "n_win": self.n_win,
            "n_payoff": self.n_payoff,
            "shared_state": self.shared_state,
            "win_schema_sha256": self.win_schema_sha256,
            "payoff_schema_sha256": self.payoff_schema_sha256,
        }


# --------------------------------------------------------------------------- #
# §11 R12 architectures                                                        #
# --------------------------------------------------------------------------- #
_WIN33 = tuple(WIN33_COLS)
_PAY8 = tuple(PAY8_COLS)

A0 = ArchSpec("A0_V1_DISJOINT", win=_WIN33, payoff=_PAY8)
A1 = ArchSpec("A1_SHARE_TO_WIN", win=SHARED41, payoff=_PAY8)
A2 = ArchSpec("A2_SHARE_TO_PAYOFF", win=_WIN33, payoff=SHARED41)
A3 = ArchSpec("A3_SHARED_BOTH", win=SHARED41, payoff=SHARED41)

R12_ARCHS: tuple[ArchSpec, ...] = (A0, A1, A2, A3)

# --------------------------------------------------------------------------- #
# §23 R13 ablations — counts frozen at 41 / 59 / 49 / 47 / 73                   #
# --------------------------------------------------------------------------- #
B0 = ArchSpec("B0_SHARED41", win=SHARED41, payoff=SHARED41)
B1 = ArchSpec("B1_PLUS_SPACE18", win=SHARED41 + SPACE18, payoff=SHARED41 + SPACE18)
B2 = ArchSpec("B2_PLUS_PATH8", win=SHARED41 + PATH8, payoff=SHARED41 + PATH8)
B3 = ArchSpec("B3_PLUS_VOL6", win=SHARED41 + VOL6, payoff=SHARED41 + VOL6)
B4 = ArchSpec(
    "B4_ALL",
    win=SHARED41 + SPACE18 + PATH8 + VOL6,
    payoff=SHARED41 + SPACE18 + PATH8 + VOL6,
)

R13_ARCHS: tuple[ArchSpec, ...] = (B0, B1, B2, B3, B4)

R13_EXPECTED_COUNTS: dict[str, int] = {
    "B0_SHARED41": 41,
    "B1_PLUS_SPACE18": 59,
    "B2_PLUS_PATH8": 49,
    "B3_PLUS_VOL6": 47,
    "B4_ALL": 73,
}

for _spec in R13_ARCHS:
    assert _spec.n_win == R13_EXPECTED_COUNTS[_spec.name], _spec.describe()
    assert _spec.n_payoff == R13_EXPECTED_COUNTS[_spec.name], _spec.describe()
    # In every B variant Win and Payoff heads consume the identical state.
    assert _spec.shared_state, _spec.name


def get_arch(name: str) -> Optional[ArchSpec]:
    for spec in tuple(R12_ARCHS) + tuple(R13_ARCHS):
        if spec.name == name or spec.name.startswith(name):
            return spec
    return None


# Columns V2 joins on / keys the frames by.
LABEL_KEYS: tuple[str, ...] = ("symbol", "decision_bar", "side", "horizon")
