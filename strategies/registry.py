from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class StrategySpec:
    name: str
    description: str
    default_params: dict
    run: Callable[[dict[str, pd.DataFrame], dict], dict]


# Explicit registry by design. Add a strategy here only after its hypothesis exists.
_REGISTRY: dict[str, StrategySpec] = {}


def register(spec: StrategySpec) -> None:
    if spec.name in _REGISTRY:
        raise ValueError(f"Duplicate strategy name: {spec.name}")
    _REGISTRY[spec.name] = spec


def all_strategies() -> dict[str, StrategySpec]:
    return dict(_REGISTRY)
