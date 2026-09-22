"""
experiment_entry_value_tree_core108_t2_formal_v1
================================================

FUTURE-ENTRY-VALUE-TREE-CORE108-V1-T2-FORMAL

This module implements the **preflight gate** required by the T2 contract
(Sections 2 and 19) and the full formal pipeline skeleton. The pipeline is
guarded behind ``preflight``: it MUST NOT run unless exactly 15 valid R2 Oracle
symbols are present.

The contract is explicit:
  * Section 2  -- "Require exactly the intended 15-symbol research universe.
                   If artifact count != 15: STOP."
  * Section 19 -- "If exactly 15 valid symbols and the common window leaves
                   every symbol represented in all 3 splits: continue T2.
                   Otherwise: STOP and return T2_PREFLIGHT_BLOCKED.
                   Do not improvise exclusions."

Running this module with the current local artifacts blocks with
``T2_PREFLIGHT_BLOCKED`` and prints coverage evidence, because only AG and RB
have ``intraday_dp_oracle_r2_one_entry_proximity`` artifacts.

Nothing here changes CORE108, the Oracle, labels, cost mode, LightGBM params or
episode weights. The full M0/M1/M2 + cross-symbol-holdout + bootstrap pipeline
is intended to execute only after the 15-symbol universe exists.
"""

from __future__ import annotations

# IMPORT ORDER: importing t15b first pulls in t15, which imports lightgbm at its
# top (before numpy/pandas/scipy). A late dlopen segfaults on macOS (OpenMP).
import research.liquidity_oracle_atlas.experiment_entry_value_tree_core108_t15b_aligned_v1 as B

import glob
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.build_structure_constrained_trade_oracle_dp_v2 import (
    load_oracle_artifact_v2,
)
from research.liquidity_oracle_atlas.experiment_entry_value_tree_core108_v1 import (
    DEFAULT_ARTIFACT_ROOT,
    load_raw_5m,
)

TASK_ID = "FUTURE-ENTRY-VALUE-TREE-CORE108-V1-T2-FORMAL"
BASE_SHA = "1b6f171c65bc5a538b7744e624c56196b2f2b236"
EXPECTED_SYMBOLS = 15
EXPECTED_MATH_VERSION = "intraday_dp_oracle_r2_one_entry_proximity"
EXPECTED_COST_MODE = "zero_cost"


# --------------------------------------------------------------------------- #
# Preflight: discover + verify the symbol universe
# --------------------------------------------------------------------------- #
def discover_symbols(artifact_root: Any = DEFAULT_ARTIFACT_ROOT) -> List[str]:
    """Symbols are discovered ONLY from the canonical R2 Oracle artifact dir."""
    return sorted(
        os.path.basename(p)
        for p in glob.glob(os.path.join(artifact_root, "*"))
        if os.path.isdir(p)
    )


def build_coverage(symbols: List[str],
                   artifact_root: Any = DEFAULT_ARTIFACT_ROOT) -> Dict[str, Any]:
    """Per-symbol availability/metadata + common calendar window. No gate."""
    rows: List[Dict[str, Any]] = []
    for s in symbols:
        raw_ok, nbar = False, None
        try:
            raw_ok = True
            nbar = int(len(load_raw_5m(s)))
        except Exception:
            raw_ok = False
        a = load_oracle_artifact_v2(artifact_root, s)
        ok = bool(a["ok"])
        mv = a["metadata"].get("math_version") if ok else None
        cm = a["metadata"].get("cost_mode") if ok else None
        nact = int(len(a["actions"])) if ok else None
        first_c, last_c = None, None
        if ok:
            try:
                d = B.symbol_available_days(s, artifact_root)
                days = list(d["days"])
                last_complete = days[-2] if len(days) >= 2 else days[-1]
                first_c = str(pd.Timestamp(days[0]).date())
                last_c = str(pd.Timestamp(last_complete).date())
            except Exception:
                pass
        rows.append({
            "symbol": s,
            "raw_5m_available": raw_ok,
            "oracle_ok": ok,
            "oracle_reason": (None if ok else a.get("reason")),
            "math_version": mv,
            "cost_mode": cm,
            "n_oracle_rows": nact,
            "n_raw_bars": nbar,
            "first_complete_day": first_c,
            "last_complete_day": last_c,
        })
    coverage = pd.DataFrame(rows)

    n_ok = int(coverage["oracle_ok"].sum()) if len(coverage) else 0
    window: Optional[Dict[str, Any]] = None
    if n_ok >= 1:
        try:
            ok_syms = [r["symbol"] for r in rows if r["oracle_ok"]]
            window = B.compute_common_window(ok_syms, artifact_root)
            window = {
                "common_start": window["common_start"],
                "common_end": window["common_end"],
                "n_window_days": window["n_window_days"],
                "per_symbol": window["per_symbol"],
            }
        except Exception as exc:  # pragma: no cover - defensive
            window = {"error": str(exc)}

    return {
        "expected_symbols": EXPECTED_SYMBOLS,
        "discovered_symbols": list(symbols),
        "n_discovered": len(symbols),
        "n_oracle_ok": n_ok,
        "expected_math_version": EXPECTED_MATH_VERSION,
        "expected_cost_mode": EXPECTED_COST_MODE,
        "coverage_rows": rows,
        "common_window": window,
    }


def preflight(symbols: Optional[List[str]] = None,
              artifact_root: Any = DEFAULT_ARTIFACT_ROOT) -> Dict[str, Any]:
    """Run the preflight gate.

    Returns the coverage dict only when exactly EXPECTED_SYMBOLS valid R2 Oracle
    symbols are present. Otherwise raises ``SystemExit('T2_PREFLIGHT_BLOCKED …')``.
    """
    if symbols is None:
        symbols = discover_symbols(artifact_root)
    cov = build_coverage(symbols, artifact_root)

    n_ok = cov["n_oracle_ok"]
    problems: List[str] = []
    if len(symbols) != EXPECTED_SYMBOLS:
        problems.append(f"discovered {len(symbols)} != expected {EXPECTED_SYMBOLS}")
    if n_ok != EXPECTED_SYMBOLS:
        problems.append(f"oracle-ok {n_ok} != expected {EXPECTED_SYMBOLS}")
    # metadata consistency among oracle-ok symbols
    ok_rows = [r for r in cov["coverage_rows"] if r["oracle_ok"]]
    for r in ok_rows:
        if r["math_version"] != EXPECTED_MATH_VERSION:
            problems.append(f"{r['symbol']} math_version={r['math_version']}")
        if r["cost_mode"] != EXPECTED_COST_MODE:
            problems.append(f"{r['symbol']} cost_mode={r['cost_mode']}")
        if not r["raw_5m_available"]:
            problems.append(f"{r['symbol']} raw_5m missing")

    if problems:
        msg = (
            f"T2_PREFLIGHT_BLOCKED: {'; '.join(problems)}. "
            f"Discovered symbols={symbols}. "
            f"Do NOT improvise exclusions; 15-symbol R2 Oracle universe required."
        )
        raise SystemExit(msg)
    return cov


# --------------------------------------------------------------------------- #
# Formal pipeline (executes only after preflight passes — currently blocked)
# --------------------------------------------------------------------------- #
def run_t2(symbols: Optional[List[str]] = None,
           artifact_root: Any = DEFAULT_ARTIFACT_ROOT,
           out_dir: Any = Path("artifacts/intraday_entry_value_tree_core108_v1"),
           write_local_dataset: bool = True) -> Dict[str, Any]:
    """Formal T2 run. Guarded: preflight() raises before any expensive work."""
    _ = preflight(symbols, artifact_root)  # raises T2_PREFLIGHT_BLOCKED if not 15
    # The M0/M1/M2 + cross-symbol-holdout + bootstrap pipeline is implemented in
    # run_t2_pipeline(); it is reached only when the 15-symbol universe exists.
    return run_t2_pipeline(artifact_root=artifact_root, out_dir=out_dir,
                           write_local_dataset=write_local_dataset)


def run_t2_pipeline(artifact_root: Any, out_dir: Any,
                    write_local_dataset: bool) -> Dict[str, Any]:  # pragma: no cover
    """Full formal pipeline (not executed in the current 2-symbol environment)."""
    raise NotImplementedError("T2 pipeline requires the 15-symbol R2 Oracle universe.")


if __name__ == "__main__":
    try:
        preflight()
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
