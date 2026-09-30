"""dp_label_structural_v1
=========================

Read-only Streamlit viewer logic for the canonical structural DP labels
(FUT-M15-STRUCTURAL-DP-LABEL-V1).

The Viewer ONLY reads the canonical artifact
``artifacts/structural_dp_labels_m15_v1/<symbol>/structural_dp_labels.parquet``
produced by ``build_structural_dp_labels_m15_v1``. It never recomputes
best_entry_gap_atr / tp_atr / remaining_target_atr / target. It draws the
candidate SR/LIQ zones, the structural target zone, and the entry / TP /
target price levels so the three structural quantities are visually auditable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from research.liquidity_oracle_atlas.build_structural_dp_labels_m15_v1 import (
    ARTIFACT_ROOT_DIRNAME,
    load_structural_labels,
)


def load_structural_labels_for_symbol(symbol: str) -> Optional[pd.DataFrame]:
    res = load_structural_labels(Path("artifacts") / ARTIFACT_ROOT_DIRNAME, symbol)
    return res["df"] if res.get("ok") else None


def select_label_near(df: pd.DataFrame, entry_fill_index: int) -> Optional[pd.Series]:
    if df is None or len(df) == 0:
        return None
    diff = (df["entry_fill_index"].to_numpy(int) - int(entry_fill_index)).abs()
    return df.iloc[diff.argmin()]


def previous_five(df: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    d2 = df.sort_values("entry_fill_time").reset_index(drop=True)
    pos = d2.index[d2["label_id"] == row["label_id"]]
    if len(pos) == 0:
        return pd.DataFrame()
    i = int(pos[0])
    return d2.iloc[max(0, i - 5):i]  # up to five PRIOR, completed labels


def find_structural_representative_cases(df: pd.DataFrame) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}

    def _pick(mask, sortcol, asc, n=3):
        sub = df[mask]
        if sub.empty:
            return []
        return sub.sort_values(sortcol, ascending=asc)["label_id"].head(n).tolist()

    out["gap_zero"] = _pick(df["best_entry_gap_atr"] < 0.02, "best_entry_gap_atr", True)
    out["gap_large"] = _pick(df["best_entry_gap_atr"].notna(), "best_entry_gap_atr", False)
    out["early_tp_large_remaining"] = _pick(
        (df["remaining_target_atr"] > 1e-9) & df["target_price"].notna(),
        "remaining_target_atr", False,
    )
    out["near_target_early_tp"] = _pick(
        (df["remaining_target_atr"] > 1e-9) & (df["remaining_target_atr"] < 0.3)
        & df["target_price"].notna(),
        "remaining_target_atr", True,
    )
    out["target_reached"] = _pick(df["target_reached"] == 1, "best_entry_gap_atr", True)
    # same-direction continuation: next trade same direction
    dd = df.sort_values("entry_fill_time").reset_index(drop=True)
    same_dir: List[str] = []
    for i in range(1, len(dd)):
        if dd.loc[i, "direction"] == dd.loc[i - 1, "direction"]:
            same_dir.append(dd.loc[i, "label_id"])
    out["same_direction_continuation"] = same_dir[:3]
    return out


# --------------------------------------------------------------------------- #
# Overlay drawing (pure geometry on the existing 15m Plotly fig)
# --------------------------------------------------------------------------- #
def _parse_zone(z: str):
    b, t = z.split(":")
    return float(b), float(t)


def draw_structural_overlay(
    fig: Any,
    row: pd.Series,
    band_lo: int,
    band_hi: int,
) -> None:
    """Draw candidate SR/LIQ zones, target zone, and entry/TP/target levels.

    band_lo / band_hi are execution-bar indices spanning the candidate region.
    No quantity is recomputed here; all prices come from the canonical row.
    """
    x0 = float(band_lo) - 0.5
    x1 = float(band_hi) + 0.5

    # candidate structure zones
    zones = str(row.get("candidate_structure_zones", "") or "").split(";")
    types = str(row.get("candidate_structure_types", "") or "").split(";")
    for j, z in enumerate(zones):
        if not z:
            continue
        b, t = _parse_zone(z)
        typ = types[j] if j < len(types) else ""
        is_liq = "LIQUIDITY" in typ or "LIQ" in typ
        fill = "rgba(255,140,0,0.12)" if is_liq else "rgba(255,193,7,0.12)"
        fig.add_shape(
            type="rect", xref="x", yref="y", x0=x0, x1=x1, y0=b, y1=t,
            fillcolor=fill, line={"width": 0}, layer="below",
        )

    # structural target zone
    tb = row.get("target_zone_bottom")
    tt = row.get("target_zone_top")
    if pd.notna(tb) and pd.notna(tt):
        fig.add_shape(
            type="rect", xref="x", yref="y", x0=x0, x1=x1, y0=float(tb), y1=float(tt),
            fillcolor="rgba(0,200,120,0.18)", line={"width": 0}, layer="below",
        )

    entry = float(row["entry_fill_price"])
    tp = float(row["tp_price"])
    tgt = float(row["target_price"]) if pd.notna(row.get("target_price")) else None

    # entry vertical marker
    fig.add_shape(
        type="line", xref="x", yref="y", x0=row["entry_fill_index"], x1=row["entry_fill_index"],
        y0=entry, y1=entry, line={"color": "rgba(255,209,102,1)", "width": 2}, layer="above",
    )
    # entry / TP / target horizontal levels
    fig.add_shape(
        type="line", xref="x", yref="y", x0=x0, x1=x1, y0=entry, y1=entry,
        line={"color": "rgba(255,209,102,0.9)", "width": 1.5}, layer="above",
    )
    fig.add_shape(
        type="line", xref="x", yref="y", x0=x0, x1=x1, y0=tp, y1=tp,
        line={"color": "rgba(0,220,255,0.95)", "width": 2}, layer="above",
    )
    if tgt is not None:
        fig.add_shape(
            type="line", xref="x", yref="y", x0=x0, x1=x1, y0=tgt, y1=tgt,
            line={"color": "rgba(0,200,120,0.95)", "width": 2, "dash": "dash"}, layer="above",
        )
        # shaded "remaining" gap between TP and target
        lo_g, hi_g = sorted([tp, tgt])
        fig.add_shape(
            type="rect", xref="x", yref="y", x0=x0, x1=x1, y0=lo_g, y1=hi_g,
            fillcolor="rgba(120,120,120,0.18)", line={"width": 0}, layer="below",
        )

    # annotations
    fig.add_annotation(
        x=row["entry_fill_index"], y=entry, text="DP BEST ENTRY",
        showarrow=True, arrowhead=2, ax=0, ay=-26,
        font={"size": 9, "color": "#FFD166"}, opacity=0.95,
    )
    fig.add_annotation(
        x=x1, y=tp, text=f"TP  tp_atr={float(row['tp_atr']):.3f}",
        showarrow=False, xanchor="left", font={"size": 9, "color": "#00DCFf"},
    )
    if tgt is not None:
        fig.add_annotation(
            x=x1, y=tgt,
            text=f"TARGET  remaining_atr={float(row['remaining_target_atr']):.3f}",
            showarrow=False, xanchor="left", font={"size": 9, "color": "#00C878"},
        )


def structural_metrics(row: pd.Series) -> List[tuple]:
    return [
        ("direction", row["direction"]),
        ("best_entry_gap_atr", round(float(row["best_entry_gap_atr"]), 4)),
        ("tp_atr", round(float(row["tp_atr"]), 4)),
        ("remaining_target_atr", round(float(row["remaining_target_atr"]), 4)),
        ("target_type", row.get("target_structure_type")),
        ("target_tf", row.get("target_structure_timeframe")),
        ("target_price", round(float(row["target_price"]), 3)
         if pd.notna(row.get("target_price")) else None),
        ("target_reached", bool(row.get("target_reached", 0) == 1)),
        ("exit_reason", row.get("exit_reason")),
        ("candidate_structures", int(row.get("candidate_structure_count", 0))),
        ("atr_owner", row.get("atr_owner")),
    ]
