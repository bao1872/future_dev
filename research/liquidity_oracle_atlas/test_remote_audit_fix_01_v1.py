"""test_remote_audit_fix_01_v1
================================

REMOTE-AUDIT-FIX-01 T0/T1 tests (remote audit of base 5d6f6a1).

Each blocker raised remotely is guarded by a test here, so it cannot silently
come back:

A. VIEWER SCHEMA -- the ONLY canonical candidate-time column is
   ``candidate_time``. Guarded statically (every column the viewer reads must be
   a column the offline builder actually writes) AND end-to-end (a
   schema-complete fixture artifact driven through the real chain
   load_oracle -> _build_fig -> _meta_table, never a partial smoke test).

B. NO PERMANENT BLACKLIST -- a structure that produced no canonical trade at
   decision bar t MUST be reconsidered at t+k. Forward progress is
   DECISION-level (scan_pos = t + 1); no structure_id is ever banned forever.

C. CO-PRESENT OWNER -- when several structures sit on one bar, the candidate is
   the canonical primary structure (nearest by bar_zone_distance), NOT the
   lexicographically smallest structure_id.

D. Hot-path differential -- the vectorized target-touch must equal the naive
   reverse-scan implementation bit-for-bit (it must stay a pure speedup).
"""

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.liquidity_oracle_atlas.structural_god_oracle_m15_v4 import (
    _next_target_touch,
    _scan_next_candidate,
)
from research.liquidity_oracle_atlas.structural_event_dp_kernel_v3 import (
    market_view_from_arrays,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
VIEWER_PATH = REPO_ROOT / "pages" / "6_Indicator_Viewer.py"
BUILDER_PATH = (
    REPO_ROOT
    / "research"
    / "liquidity_oracle_atlas"
    / "build_god_oracle_latest_artifact_v1.py"
)
REAL_ARTIFACT_DIR = REPO_ROOT / "artifacts" / "god_oracle_m15_latest"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_mv(opens, highs, lows):
    n = len(opens)
    closes = list(opens)
    times = [
        pd.Timestamp("2020-01-01") + pd.Timedelta(minutes=15 * i) for i in range(n)
    ]
    return market_view_from_arrays(
        opens, highs, lows, closes, times=times,
        unit_starts=np.array([0], dtype=np.int64), symbol="SYNTH",
    )


PROD_PATH = (
    REPO_ROOT
    / "research"
    / "liquidity_oracle_atlas"
    / "structural_god_oracle_m15_v4.py"
)


def test_production_never_imports_reference():
    """TP counter `reference_call_count` is 0 on the production path.

    The production module must not import (hence cannot call into) the
    independent O(L^2) reference; parity is only ever established from OUTSIDE,
    by the differential tests.
    """
    for line in PROD_PATH.read_text().splitlines():
        if line.startswith(("import ", "from ")):
            assert "structural_god_oracle_reference_m15_v4" not in line, line


def _per_bar_single(n, bars, sid, bottom=99.0, top=100.0):
    pb = [None] * n
    for t in bars:
        pb[t] = [(sid, "SR", bottom, top)]
    return pb


# --------------------------------------------------------------------------- #
# B. failed decision must NOT blacklist the structure
# --------------------------------------------------------------------------- #
def test_failed_decision_does_not_blacklist_structure():
    """A structure that failed at decision bar t IS reconsidered at t+k.

    The wrong behavior was: no canonical trade -> structure_id added to a
    permanent skip set -> never a candidate again. Forward progress must be
    decision-level only (scan_pos = t + 1). ``_scan_next_candidate`` carries no
    cross-call memory, so re-scanning past t must find the same sid again.
    """
    n = 40
    S = "SR|m15|0|10|100.0|99.0|5.0"
    per_bar = _per_bar_single(n, (10, 20), S)
    lows = [0.0] * n
    highs = [0.0] * n

    c1, sid1 = _scan_next_candidate(per_bar, 0, {S}, n, lows, highs)
    assert (c1, sid1) == (10, S)

    # decisions-level forward progress after a failed decision at bar 10
    scan_pos = c1 + 1
    c2, sid2 = _scan_next_candidate(per_bar, scan_pos, {S}, n, lows, highs)
    assert (c2, sid2) == (20, S), (
        "structure was blacklisted after one failed decision "
        "(permanent no-trade skip set still present)"
    )


# --------------------------------------------------------------------------- #
# C. co-present structures owned by primary_structure (nearest), not string order
# --------------------------------------------------------------------------- #
def test_co_present_structures_use_primary_owner():
    """The nearest co-present structure wins, NOT sorted(structure_id)[0]."""
    n = 40
    S_LEX_FIRST_FAR = "AAA|m15|0|10|1.0|0.0|1.0"   # lexicographically smallest
    S_LEX_LAST_NEAR = "ZZZ|m15|0|10|10.5|10.0|1.0"  # nearest to this bar
    per_bar = [None] * n
    per_bar[10] = [
        (S_LEX_FIRST_FAR, "SR", 0.0, 1.0),    # bar is ~9.2 away
        (S_LEX_LAST_NEAR, "SR", 10.0, 10.5),  # bar is INSIDE this zone -> d = 0
    ]
    lows = [0.0] * n
    highs = [0.0] * n
    lows[10], highs[10] = 10.2, 10.4

    eligible = {S_LEX_FIRST_FAR, S_LEX_LAST_NEAR}
    assert sorted(eligible)[0] == S_LEX_FIRST_FAR  # what lexical order would pick

    c, sid = _scan_next_candidate(per_bar, 0, eligible, n, lows, highs)
    assert c == 10
    assert sid == S_LEX_LAST_NEAR, "co-present owner must be the NEAREST structure"
    assert sid != sorted(eligible)[0], "owner fell back to lexical structure_id order"


# --------------------------------------------------------------------------- #
# D. vectorized target-touch must equal the naive reference
# --------------------------------------------------------------------------- #
def _naive_next_target_touch(mv, lo, hi, direction, target_price):
    """The original reverse-scan implementation, kept verbatim as reference."""
    n = mv.n
    nt = np.full(n, -1, dtype=np.int64)
    if target_price is None:
        return nt
    tp = float(target_price)
    highs, lows = mv.highs, mv.lows
    nxt = -1
    for t in range(int(hi), int(lo) - 1, -1):
        touched = (float(highs[t]) >= tp) if direction == "LONG" else (float(lows[t]) <= tp)
        if touched:
            nxt = t
        nt[t] = nxt
    return nt


def test_next_target_touch_matches_naive_reference():
    """The vectorization is a pure speedup: identical arrays, bit-for-bit."""
    rng = np.random.RandomState(20261003)
    n = 400
    highs = 100.0 + np.cumsum(rng.randn(n) * 0.05)
    lows = highs - np.abs(rng.randn(n)) * 0.05
    opens = highs - 0.01
    mv = make_mv(opens.tolist(), highs.tolist(), lows.tolist())

    cases = [
        ("LONG", float(highs.mean())),          # touched often
        ("LONG", float(highs.max() + 10.0)),    # never touched -> all -1
        ("SHORT", float(lows.mean())),          # touched often
        ("SHORT", float(lows.min() - 10.0)),    # never touched -> all -1
    ]
    for lo, hi in ((0, n - 1), (7, 250), (300, n - 1)):
        for direction, tp in cases:
            got = _next_target_touch(mv, lo, hi, direction, tp)
            exp = _naive_next_target_touch(mv, lo, hi, direction, tp)
            assert np.array_equal(got, exp), f"mismatch at lo={lo} hi={hi} {direction}"


# --------------------------------------------------------------------------- #
# A. viewer artifact schema + FULL chain
# --------------------------------------------------------------------------- #
_VIEWER_COL_RE = re.compile(r"\b(?:sel|cr)\[\s*['\"]([a-z_]+)['\"]\s*\]")


def _viewer_read_columns() -> set:
    return set(_VIEWER_COL_RE.findall(VIEWER_PATH.read_text()))


def _builder_trade_columns() -> set:
    """Column names actually written into trades.parquet by the builder."""
    src = BUILDER_PATH.read_text()
    block = src[
        src.index("trade_rows.append({"): src.index("trades_df = pd.DataFrame")
    ]
    keys = set(re.findall(r'"([a-z_]+)":\s', block))
    assert keys, "static extraction of the builder trade schema failed"
    return keys


def test_viewer_columns_are_subset_of_artifact_schema():
    """Statically kills the KeyError class of bug (candidate_start_time)."""
    reads = _viewer_read_columns()
    written = _builder_trade_columns()
    missing = reads - written
    assert not missing, (
        f"viewer reads columns the artifact never writes: {sorted(missing)}; "
        f"artifact schema = {sorted(written)}"
    )
    assert reads, "static extraction of the viewer read-set failed"


def test_canonical_candidate_time_name_only():
    """ONE canonical name: candidate_time. The alias must never come back."""
    written = _builder_trade_columns()
    assert "candidate_time" in written
    assert "candidate_start_time" not in written
    assert "candidate_start_time" not in _viewer_read_columns()


FIXTURE_N = 60


def _write_fixture_artifact(root: Path, columns) -> None:
    root.mkdir(parents=True, exist_ok=True)
    base = pd.Timestamp("2020-01-01 09:00:00")
    times = [base + pd.Timedelta(minutes=15 * i) for i in range(FIXTURE_N)]
    bars = pd.DataFrame({
        "bar_start_time": times,
        "open": [100.0] * FIXTURE_N,
        "high": [101.0] * FIXTURE_N,
        "low": [99.0] * FIXTURE_N,
        "close": [100.5] * FIXTURE_N,
    })
    trade = {
        "trade_seq": 1,
        "event_id": 7,
        "structure_id": "SR|m15|0|10|100.0|99.0|5.0",
        "candidate_decision_index": 10,
        "candidate_time": times[10],
        "zone_bottom": 99.0,
        "zone_top": 100.0,
        "oracle_direction": "LONG",
        "best_entry_decision_index": 10,
        "best_entry_fill_index": 11,
        "best_entry_fill_time": times[11],
        "best_entry_price": 100.0,
        "best_entry_gap_atr": 0.1,
        "target_structure_id": "SR|m15|0|20|106.0|105.0|5.0",
        "target_structure_type": "SR",
        "target_price": 105.0,
        "exit_fill_index": 20,
        "exit_fill_time": times[20],
        "exit_price": 105.0,
        "tp_atr": 1.0,
        "exit_reason": "TARGET_TOUCH",
        "utility": 5.0,
    }
    missing = set(columns) - set(trade)
    assert not missing, f"fixture missing columns: {sorted(missing)}"
    pd.DataFrame([trade]).to_parquet(root / "trades.parquet", index=False)
    bars.to_parquet(root / "bars.parquet", index=False)
    pd.DataFrame({
        "bar_index": list(range(FIXTURE_N)),
        "geom": ["{}"] * FIXTURE_N,
    }).to_parquet(root / "structures.parquet", index=False)
    (root / "manifest.json").write_text(json.dumps({
        "source_git_sha": "FIXTURE",
        "generated_at": "1970-01-01T00:00:00+00:00",
        "symbol": "AG",
        "canonical_trade_count": 1,
        "math_version": "structural_god_oracle_m15_v4",
        "oracle_meta": {},
    }))


def _import_viewer():
    spec = importlib.util.spec_from_file_location(
        "god_oracle_indicator_viewer", VIEWER_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_viewer_full_chain_load_figure_meta_table(tmp_path):
    """FULL chain on a schema-complete fixture: load -> figure -> meta table."""
    viewer = _import_viewer()
    _write_fixture_artifact(tmp_path, sorted(_builder_trade_columns()))
    viewer._ARTIFACT_DIR = tmp_path

    data = viewer.load_oracle("AG")           # 1) load
    assert data["canonical_trade_count"] == 1
    sel = data["canon"][0]

    fig = viewer._build_fig(data, sel)        # 2) figure
    assert len(fig.data) > 0

    table = viewer._meta_table(sel)           # 3) meta table (KeyError zone)
    fields = set(table["field"])
    assert "candidate_time" in fields
    assert pd.Timestamp(table.loc[table["field"] == "candidate_time", "value"].iloc[0])


def test_viewer_full_chain_on_real_artifact_if_present():
    """Same full chain against the materialized AG artifact, when present."""
    if not (REAL_ARTIFACT_DIR / "trades.parquet").exists():
        pytest.skip("latest artifact not materialized locally")
    viewer = _import_viewer()
    data = viewer.load_oracle("AG")
    assert data["canon"], "artifact has no canonical trades"

    # every canonical trade must survive the whole render chain
    for sel in data["canon"]:
        assert len(viewer._build_fig(data, sel).data) > 0
        assert "candidate_time" in set(viewer._meta_table(sel)["field"])


# --------------------------------------------------------------------------- #
# B (orchestration level): the loop itself must not blacklist anything
# --------------------------------------------------------------------------- #
def test_failed_decision_reconsidered_later_end_to_end():
    """Orchestration-level guard for B on the real AG stream.

    The unit test above only proves ``_scan_next_candidate`` is stateless; the
    blacklist actually lived in the CALLER loop (eligible_sids - no_trade_sids),
    so it is invisible to that unit test. Here every candidate decision is
    recorded with a spy on ``pick_target_fn``, then we assert that at least one
    structure_id which produced NO canonical trade at decision bar t is
    ATTEMPTED AGAIN at a later bar t+k. With a permanent skip set that count is
    necessarily 0, so this assertion cannot pass vacuously.
    """
    from research.liquidity_oracle_atlas.structural_god_oracle_m15_v4 import (
        _run_god_oracle_core,
        pick_directional_target_v4,
        solve_direction_god_v4,
    )

    attempts = []

    def spy_pick(geom_prev, direction, zb, zt, tf, seg, i, sr_fs, cand_sid):
        attempts.append((int(i), str(cand_sid)))
        return pick_directional_target_v4(
            geom_prev, direction, zb, zt, tf, seg, i, sr_fs, cand_sid
        )

    res = _run_god_oracle_core(
        "AG", None, None,
        solve_direction_fn=solve_direction_god_v4,
        pick_target_fn=spy_pick,
    )
    canon = [r for r in res["records"] if r["canonical_oracle_trade"]]
    success = {
        (int(r["candidate_start_bar"]), str(r["structure_id"])) for r in canon
    }
    failed = [a for a in set(attempts) if a not in success]
    failed_by_sid: dict = {}
    for b, s in failed:
        failed_by_sid.setdefault(s, []).append(b)

    reconsidered = sum(
        1 for b2, s2 in set(attempts)
        if any(b1 < b2 for b1 in failed_by_sid.get(s2, ()))
    )
    assert failed, "no failed candidate decision on AG -- this test is vacuous"
    assert reconsidered > 0, (
        "structures that failed a decision are never reconsidered later "
        "(a permanent no-trade blacklist has been reintroduced)"
    )
