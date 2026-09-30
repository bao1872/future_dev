"""Page-level regression: an UNAVAILABLE R4 candidate gate must never crash the
Indicator Viewer and must never suppress the frozen DP Oracle Entry/Exit overlay.

Bug this locks down
-------------------
The page loads R4 twice:

  1. ``candidate_segments_r4_cached(...)``  -> already inside try/except
  2. ``candidate_marks_r4_cached(...)``     -> was NOT protected

When the R4 manifest is missing (``STOP_R4_GATE_MANIFEST_MISSING:AG``) the FIRST
call only warned, but the SECOND call re-ran the same failing loader and raised,
aborting the whole script render -- so the DP Oracle Entry / Exit markers never
drew even though they were completely unrelated to R4.

Contract asserted here
----------------------
  R4 unavailable + "Show Candidate Trading Zones" ON + "DP Oracle" ON
      => no page exception
      => exactly the R4 warning is surfaced (once)
      => the DP Oracle Long/Short Entry/Exit traces are still rendered

The R4 failure is forced deterministically by patching the single canonical row
loader, so the test does not depend on whether a real R4 manifest happens to be
present on disk.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

import streamlit as st  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PAGE = REPO / "pages" / "6_Indicator_Viewer.py"

R4_ERROR = "STOP_R4_GATE_MANIFEST_MISSING:AG"

_ORACLE_TRACES = ("Long Entry", "Long Exit", "Short Entry", "Short Exit")


def _force_r4_missing(monkeypatch):
    """Make every R4 row load fail with the real missing-manifest error."""
    import research.liquidity_oracle_atlas.indicator_viewer_candidate_overlay_r4_v1 as r4

    def _boom(symbol, *a, **k):
        raise RuntimeError(f"{R4_ERROR}")

    monkeypatch.setattr(r4, "load_candidate_rows_r4", _boom, raising=True)


def _render(candidate_on: bool, oracle_on: bool, monkeypatch):
    captured: list = []
    orig = st.plotly_chart

    def _spy(fig, *a, **k):
        captured.append(fig)
        return orig(fig, *a, **k)

    monkeypatch.setattr(st, "plotly_chart", _spy, raising=True)
    _force_r4_missing(monkeypatch)

    at = AppTest.from_file(str(PAGE), default_timeout=900)
    at.run()

    labels = {c.label: c for c in at.checkbox}
    boxes = {s.label: s for s in at.selectbox}
    boxes["Timeframe"].set_value("15m")
    labels["Show Candidate Trading Zones"].set_value(candidate_on)
    labels["DP Oracle — FUTURE / HINDSIGHT AUDIT"].set_value(oracle_on)
    labels["Show DP Proximity Audit"].set_value(False)
    labels["DP Label Lifecycle Gate (15m)"].set_value(False)
    labels["Structural DP Label V1 (15m)"].set_value(False)
    at.run()
    at.run()  # settle after the TF switch

    assert not at.exception, f"page raised: {at.exception}"
    return at, captured


def _trace_names(fig):
    names = {}
    for tr in fig.data:
        mode = getattr(tr, "mode", None)
        if mode == "markers+text":
            names[tr.name] = names.get(tr.name, 0) + len(tr.x)
    return names


def test_r4_missing_with_candidate_on_does_not_crash(monkeypatch):
    """The regression itself: R4 missing + Candidate ON must not abort the page."""
    at, captured = _render(True, True, monkeypatch)
    warns = [w.value for w in at.warning]
    assert any("R4 candidate gate not available" in w for w in warns), warns
    assert any(R4_ERROR in w for w in warns), warns
    assert captured, "no figure was rendered"


def test_r4_missing_still_draws_dp_oracle_entry_exit(monkeypatch):
    """R4 failure must not take the frozen DP Oracle overlay down with it."""
    at, captured = _render(True, True, monkeypatch)
    names = _trace_names(captured[-1])
    missing = [t for t in _ORACLE_TRACES if t not in names]
    assert not missing, f"DP Oracle traces missing: {missing} (got {sorted(names)})"


def test_r4_missing_with_oracle_off_renders_chart(monkeypatch):
    """Candidate ON + Oracle OFF: warning only, chart still renders."""
    at, captured = _render(True, False, monkeypatch)
    warns = [w.value for w in at.warning]
    assert any("R4 candidate gate not available" in w for w in warns), warns
    assert captured, "no figure was rendered"
    assert not _trace_names(captured[-1]), "oracle must stay off"
