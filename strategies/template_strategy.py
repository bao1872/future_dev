"""Strategy template. This module is intentionally NOT registered by default."""

from __future__ import annotations

import pandas as pd

NAME = "template_strategy"
DESCRIPTION = "Template only; copy this module when a real hypothesis is defined."
DEFAULT_PARAMS: dict = {}


def run(data: dict[str, pd.DataFrame], params: dict) -> dict:
    raise NotImplementedError(
        "Define one concrete trading hypothesis before registering this strategy."
    )
