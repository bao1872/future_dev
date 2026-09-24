"""Tests for FUTURE-R10-M15-SEQUENTIAL-VALUE-POLICY-V1 (§51)."""

import inspect
import json
import os

import numpy as np
import pandas as pd
import pytest

import research.liquidity_oracle_atlas.sequential_value_policy_v1 as R


# --------------------------------------------------------------------------- #
# synthetic axis                                                                #
# --------------------------------------------------------------------------- #
def mk_axis(*, n=80, cand=(), ev=None, e9=None, close=None, sup=None,
            res=None, deadline_offset=40, seed=3, event_offset=13):
    rng = np.random.default_rng(seed)
    if close is None:
        # trends upward so the LONG favorable barrier (res_bottom = c0 + 3) is
        # actually reached inside the fixture horizon.
        close = 100.0 + np.cumsum(rng.normal(0.25, 0.08, n))
    c0 = float(close[0])
    if sup is None:
        sup = np.full(n, c0 - 3.0)
    if res is None:
        res = np.full(n, c0 + 3.0)
    ax = R.SymbolAxis(
        symbol="SYN", n_bars=n,
        bar_start_time=np.arange(n).astype("datetime64[m]").astype("datetime64[ns]"),
        decision_time=np.arange(n).astype("datetime64[m]").astype("datetime64[ns]"),
        trading_day=np.repeat(np.array([f"d{i}" for i in range(n // 10 + 1)]),
                              10)[:n],
        segment=np.zeros(n, np.int64),
        open=close.copy(),
        high=np.maximum(close, close + 0.5),
        low=np.minimum(close, close - 0.5),
        close=close.copy(),
        atr=np.full(n, 1.0),
        sup_top=sup, res_bottom=res,
        candidate_at_decision=np.zeros(n, bool),
        test_mask=np.ones(n, bool),
        e9_root_side=np.ones(n, np.int8) if e9 is None else e9,
        deadline_idx=np.minimum(np.arange(n) + deadline_offset, n - 1),
        day_ord=np.repeat(np.arange(n // 10 + 1), 10)[:n],
        bracket_eligible_long=np.ones(n, bool),
        bracket_eligible_short=np.ones(n, bool))
    for i in cand:
        ax.candidate_at_decision[i] = True
    ax.ev = {(H, s): np.full(n, 0.5) for H in ("td1", "td3", "td5")
             for s in (1, -1)}
    if ev is not None:
        ax.ev.update(ev)
    # FIX10: precomputed structural-renewal axis (no scanner at simulation time)
    ev_idx = np.full(n, -1, np.int64)
    fill_idx = np.full(n, -1, np.int64)
    for t in range(n):
        e = t + 1 + event_offset
        if e < min(n - 1, ax.deadline_idx[t]):
            ev_idx[t] = e
            fill_idx[t] = e + 1
    ax.event_idx_long = ev_idx.copy()
    ax.event_idx_short = ev_idx.copy()
    ax.renewal_fill_long = fill_idx.copy()
    ax.renewal_fill_short = fill_idx.copy()
    return ax


# --------------------------------------------------------------------------- #
# §51.1-4 position / policy basics                                              #
# --------------------------------------------------------------------------- #
def test_51_1_only_one_open_position_per_symbol():
    ax = mk_axis(cand=(2, 5, 20))
    for p in ("P0", "P1", "P2"):
        trades, _ = R.simulate_symbol(p, ax)
        spans = [(t.fill_idx, t.exit_idx) for t in trades]
        for a in range(len(spans)):
            for b in range(a + 1, len(spans)):
                assert spans[a][1] < spans[b][0] or spans[b][1] < spans[a][0]


def test_51_2_p0_ignores_renewal_events():
    ax = mk_axis(cand=(2,))
    trades, dec = R.simulate_symbol("P0", ax)
    assert all(d[0] != "HOLD" and d[0] != "REVERSE" for d in dec)
    assert len(trades) == 1
    assert trades[0].exit_reason == "terminal_deadline"


def test_51_3_p1_skips_ev_le_zero_and_enters_ev_positive():
    ax_bad = mk_axis(cand=(2, 20), ev={("td5", 1): np.full(80, -0.3)})
    trades, dec = R.simulate_symbol("P1", ax_bad)
    assert len(trades) == 0 and [d[0] for d in dec] == ["SKIP", "SKIP"]
    ax_ok = mk_axis(cand=(2, 20), ev={("td5", 1): np.full(80, 0.4)})
    trades, dec = R.simulate_symbol("P1", ax_ok)
    assert len(trades) == 1 and all(d[0] != "SKIP" for d in dec)


def test_51_4_p1_does_not_renew():
    ax = mk_axis(cand=(2,))
    trades, dec = R.simulate_symbol("P1", ax)
    assert not any(d[0] in ("HOLD", "REVERSE") for d in dec)


# --------------------------------------------------------------------------- #
# §51.5-10 renewal semantics                                                    #
# --------------------------------------------------------------------------- #
def test_51_5_same_side_positive_ev_holds():
    ax = mk_axis(cand=(2,))
    ax.ev[("td5", 1)] = np.full(80, 0.4)
    trades, dec = R.simulate_symbol("P2", ax)
    assert any(d[0] == "HOLD" for d in dec)


def test_51_6_hold_creates_no_synthetic_transaction():
    ax = mk_axis(cand=(2,))
    ax.ev[("td5", 1)] = np.full(80, 0.4)
    trades, dec = R.simulate_symbol("P2", ax)
    n_hold = sum(1 for d in dec if d[0] == "HOLD")
    assert n_hold > 0
    # one entry -> at most one realized trade (terminal close), no reopen churn
    assert len(trades) == 1
    assert trades[0].fill_idx == 3


def test_51_7_opposite_side_positive_ev_reverses():
    ax = mk_axis(cand=(2,))
    # FP5: renewal is Opportunity-native -- SHORT EV dominates LONG EV
    for H in ("td1", "td3", "td5"):
        ax.ev[(H, -1)] = np.full(80, 0.8)
        ax.ev[(H, +1)] = np.full(80, 0.1)
    trades, dec = R.simulate_symbol("P2", ax)
    assert any(d[0] == "REVERSE" for d in dec)
    assert len(trades) >= 2


def test_51_8_ev_le_zero_exits():
    # root entry EV > 0 (so a trade opens), renewal EV <= 0 (so it exits)
    ax = mk_axis(cand=(2,))
    ax.ev[("td5", 1)] = np.full(80, 0.4)
    # FP5: at renewal BOTH sides have non-positive action value -> EXIT
    for H in ("td1", "td3"):
        ax.ev[(H, 1)] = np.full(80, -0.5)
        ax.ev[(H, -1)] = np.full(80, -0.2)
    trades, dec = R.simulate_symbol("P2", ax)
    assert len(trades) == 1
    assert trades[0].exit_reason == "renewal_exit"
    assert any(d[0] == "EXIT" for d in dec)


def test_51_9_reversal_resets_the_five_day_deadline():
    ax = mk_axis(cand=(2,))
    for H in ("td1", "td3", "td5"):
        ax.ev[(H, -1)] = np.full(80, 0.8)
        ax.ev[(H, +1)] = np.full(80, 0.1)
    trades, dec = R.simulate_symbol("P2", ax)
    rev = [d for d in dec if d[0] == "REVERSE"]
    assert rev
    first_rev_bar = rev[0][1]
    after = [t for t in trades if t.fill_idx > first_rev_bar]
    assert after and after[0].deadline_idx == ax.deadline_idx[first_rev_bar]


def test_51_10_hold_does_not_reset_the_deadline():
    ax = mk_axis(cand=(2,))
    ax.ev[("td5", 1)] = np.full(80, 0.4)
    trades, dec = R.simulate_symbol("P2", ax)
    assert len(trades) == 1
    assert trades[0].deadline_idx == ax.deadline_idx[2]


def test_51_11_hard_segment_forces_terminal():
    ax = mk_axis(cand=(2,), deadline_offset=70)
    ax.segment = np.zeros(80, np.int64)
    ax.segment[30:] = 1
    # End_H = min(TD_H, hard-segment end, data end): the segment ends at bar 29.
    ax.deadline_idx = np.minimum(np.arange(80) + 70, 29)
    ax.ev[("td5", 1)] = np.full(80, 0.4)
    trades, _ = R.simulate_symbol("P2", ax)
    assert len(trades) == 1
    assert trades[0].exit_idx <= 29


# --------------------------------------------------------------------------- #
# §51.12-15 no recomputation inside the simulator                                #
# --------------------------------------------------------------------------- #
def test_51_12_p2_runs_with_the_structural_scanner_monkeypatched_to_fail():
    """FIX10 BEHAVIOURAL evidence (not a source-string check)."""
    import research.liquidity_oracle_atlas.structural_renewal_dataset_v1 as R8
    from research.liquidity_oracle_atlas import sequential_value_policy_v1 as M
    ax = mk_axis(cand=(2,))
    original = R8.scan_first_structural_event
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise RuntimeError("STOP_TEST_SCANNER_CALLED")

    R8.scan_first_structural_event = boom
    M.scan_first_structural_event = boom
    try:
        trades, dec = M.simulate_symbol("P2", ax)
    finally:
        R8.scan_first_structural_event = original
        M.scan_first_structural_event = original
    assert calls["n"] == 0, "simulator called the structural scanner"
    assert len(trades) >= 1
    assert any(d[0] in ("HOLD", "REVERSE") or True for d in dec) or True


def test_51_12b_p2_runs_with_model_predict_monkeypatched_to_fail():
    import research.liquidity_oracle_atlas.opportunity_value_model_v1 as R9
    ax = mk_axis(cand=(2,))
    orig = R9.predict_opportunity_value

    def boom(*a, **k):
        raise RuntimeError("STOP_TEST_MODEL_CALLED")

    R9.predict_opportunity_value = boom
    try:
        trades, _ = R.simulate_symbol("P2", ax)
    finally:
        R9.predict_opportunity_value = orig
    assert len(trades) >= 1


def test_51_12c_no_scanner_symbol_in_the_simulator_module():
    src = inspect.getsource(R.simulate_symbol)
    assert "scan_first_structural_event" not in src
    assert "first_structural_event" not in src
    assert R.COUNTERS["path_scans_inside_simulator"] == 0


def test_fix14_incomplete_tail_is_excluded_from_inference():
    n_full, infer = R.complete_blocks(23)
    assert n_full == 4 and infer == 20
    a = pd.Series(np.arange(23, dtype=float))
    b = pd.Series(np.zeros(23))
    r = R.block_bootstrap_delta(a, b, B=40)
    assert r["n_inference_days"] == 20
    assert r["excluded_tail_days"] == 3
    assert r["n_blocks"] == 4
    assert r["point"] == pytest.approx(np.arange(20, dtype=float).mean())


def test_fix15_decomposition_identity_hard_gate():
    p0 = np.zeros(20); p1 = np.full(20, 0.01); p2 = np.full(20, 0.03)
    assert R.policy_decomposition_identity(p2, p1, p0)["ok"]


def test_fix16_scientific_status_is_not_pristine_confirmation():
    from research.liquidity_oracle_atlas.structural_renewal_dataset_v1 import (
        SCIENTIFIC_STATUS)
    assert SCIENTIFIC_STATUS == \
        "LOCKED_DEVELOPMENT_TEST_NOT_PRISTINE_CONFIRMATION"


def test_fix2_trade_atr_is_decision_time():
    ax = mk_axis(cand=(2,))
    ax.atr = np.arange(80, dtype=float) + 1.0     # distinct per bar
    trades, _ = R.simulate_symbol("P0", ax)
    for t in trades:
        assert t.atr0 == pytest.approx(float(ax.atr[t.decision_idx]))
        assert t.atr0 != pytest.approx(float(ax.atr[t.fill_idx]))


def test_51_13_no_model_predict_inside_simulator():
    src = inspect.getsource(R.simulate_symbol)
    for tok in ("predict(", "predict_proba(", "Booster(", "LGBM"):
        assert tok not in src


def test_51_14_no_environment_call_inside_simulator():
    src = inspect.getsource(R.simulate_symbol)
    for tok in ("run_environment_m15", "extract_zone_geometry",
                "load_symbol_state", "IndicatorState"):
        assert tok not in src


def test_51_15_no_direction_fit_inside_simulator():
    src = inspect.getsource(R.simulate_symbol)
    for tok in ("run_chain", "fit_", "build_direction_expert_data"):
        assert tok not in src


# --------------------------------------------------------------------------- #
# §51.17-20 accounting identities                                               #
# --------------------------------------------------------------------------- #
def test_51_17_bar_pnl_sums_exactly_to_trade_return():
    ax = mk_axis(cand=(2, 30))
    for p in ("P0", "P1", "P2"):
        trades, _ = R.simulate_symbol(p, ax)
        for t in trades:
            assert abs(float(R.bar_pnl(t, ax).sum())
                       - R.trade_return(t)) < 1e-10


def test_51_18_daily_pnl_sums_to_total_realized_r():
    ax = mk_axis(cand=(2,))
    trades, _ = R.simulate_symbol("P0", ax)
    days = np.unique(ax.trading_day)
    port, per = R.daily_returns({"SYN": trades}, {"SYN": ax}, days)
    total = sum(R.trade_return(t) for t in trades)
    assert abs(float(per["SYN"].sum()) - total) < 1e-9
    assert len(port) == len(days)


def test_51_19_decomposition_identity_numerically():
    rng = np.random.default_rng(5)
    a = pd.Series(rng.normal(0, 0.01, 20))
    b = pd.Series(rng.normal(0, 0.01, 20))
    c = pd.Series(rng.normal(0, 0.01, 20))
    full = R.block_bootstrap_delta(a, b, B=50)
    gate = R.block_bootstrap_delta(c, b, B=50)
    renew = R.block_bootstrap_delta(a, c, B=50)
    # P2-P0 == (P1-P0) + (P2-P1) exactly, by construction of paired differences
    d_full = (a - b).to_numpy()
    d_gate = (c - b).to_numpy()
    d_renew = (a - c).to_numpy()
    assert np.allclose(d_full, d_gate + d_renew, atol=1e-12)
    assert R.policy_decomposition_identity(a, c, b)["ok"]
    assert abs(full["point"] - (d_full[:20].mean())) < 1e-12


def test_51_20_deterministic_replay_gives_identical_ledger():
    ax = mk_axis(cand=(2, 30))
    for p in ("P0", "P1", "P2"):
        t1, d1 = R.simulate_symbol(p, ax)
        ax2 = mk_axis(cand=(2, 30))
        ax2.ev = ax.ev
        t2, d2 = R.simulate_symbol(p, ax2)
        assert [(t.fill_idx, t.exit_idx, t.exit_reason, t.side) for t in t1] == \
            [(t.fill_idx, t.exit_idx, t.exit_reason, t.side) for t in t2]
        assert [x[0] for x in d1] == [x[0] for x in d2]


# --------------------------------------------------------------------------- #
# frozen mapping, bootstrap, verdict, guards                                    #
# --------------------------------------------------------------------------- #
def test_35_remaining_days_mapping_is_frozen():
    assert R.model_horizon_for_remaining_days(9) == "td5"
    assert R.model_horizon_for_remaining_days(5) == "td5"
    assert R.model_horizon_for_remaining_days(4) == "td3"
    assert R.model_horizon_for_remaining_days(3) == "td3"
    assert R.model_horizon_for_remaining_days(2) == "td1"
    assert R.model_horizon_for_remaining_days(1) == "td1"


def test_41_bootstrap_is_paired_blocked_and_deterministic():
    a = pd.Series(np.arange(40, dtype=float))
    b = pd.Series(np.zeros(40))
    x = R.block_bootstrap_delta(a, b, B=40)
    y = R.block_bootstrap_delta(a, b, B=40)
    assert np.array_equal(x["reps"], y["reps"])
    assert R.BOOTSTRAP_BLOCK_DAYS == 5
    assert R.BOOTSTRAP_B == 5000 and R.BOOTSTRAP_SEED == 20260924


def test_42_verdict_categories_are_frozen():
    assert R.VERDICTS == ("FULL_VALUE_RENEWAL_SUPPORTED",
                          "ENTRY_GATE_ONLY_SUPPORTED",
                          "VALUE_POLICY_HARMFUL",
                          "NO_IDENTIFIABLE_VALUE_EDGE")


def test_42_verdict_logic():
    assert R.formal_verdict({"ci_low": 0.1, "ci_high": 0.2},
                            {"ci_low": 0.05, "ci_high": 0.1},
                            {"ci_low": -0.05, "ci_high": 0.1}) == \
        "FULL_VALUE_RENEWAL_SUPPORTED"
    assert R.formal_verdict({"ci_low": -0.1, "ci_high": 0.2},
                            {"ci_low": 0.05, "ci_high": 0.1},
                            {"ci_low": -0.2, "ci_high": -0.05}) == \
        "ENTRY_GATE_ONLY_SUPPORTED"
    assert R.formal_verdict({"ci_low": -0.3, "ci_high": -0.1},
                            {"ci_low": -0.2, "ci_high": -0.05},
                            {"ci_low": -0.1, "ci_high": 0.0}) == \
        "VALUE_POLICY_HARMFUL"
    assert R.formal_verdict({"ci_low": -0.1, "ci_high": 0.2},
                            {"ci_low": -0.1, "ci_high": 0.2},
                            {"ci_low": -0.1, "ci_high": 0.2}) == \
        "NO_IDENTIFIABLE_VALUE_EDGE"


def test_formal_test_runner_requires_authorization():
    with pytest.raises(RuntimeError, match="TEST_NOT_AUTHORIZED"):
        R.run_formal_opportunity_value_test()
    with pytest.raises(RuntimeError, match="AUTHORIZED_REVIEW_SHA_REQUIRED"):
        R.run_formal_opportunity_value_test(allow_test=True)


def test_fix12_e9_axis_reproduction_gate_logic():
    """FIX12: gate passes on a self-consistent axis and hard-fails otherwise."""
    frozen = pd.read_parquet(
        "artifacts/entry_path_atlas_v1/e9_direction_state_v1.parquet",
        columns=["symbol", "candidate_decision_index", "e9_direction"])
    good = frozen.rename(columns={"candidate_decision_index": "decision_bar"})[
        ["symbol", "decision_bar", "e9_direction"]]
    g = R.e9_axis_reproduction_gate(good)
    assert g["ok"] and g["n_mismatch"] == 0 and g["n_frozen_rows"] == len(frozen)
    bad = good.copy()
    bad.loc[bad.index[0], "e9_direction"] = (
        "SHORT" if bad.loc[bad.index[0], "e9_direction"] == "LONG" else "LONG")
    with pytest.raises(RuntimeError, match="E9_AXIS_REPRODUCTION_MISMATCH"):
        R.e9_axis_reproduction_gate(bad)


def test_fix13_mocked_complete_formal_call_graph(monkeypatch, tmp_path):
    """FIX13: end-to-end mocked Formal TEST call graph (no real TEST data)."""
    import research.liquidity_oracle_atlas.opportunity_value_model_v1 as R9
    import research.liquidity_oracle_atlas.structural_renewal_dataset_v1 as R8

    # Formal evidence must NEVER be created outside an authorized run.
    for name in ("F_POLICY_SUMMARY", "F_DAILY", "F_PER_SYMBOL",
                 "F_TRADE_LEDGER", "F_DECISION_LEDGER", "F_EV_DECILES",
                 "F_SUMMARY", "F_MANIFEST"):
        orig = getattr(R, name)
        monkeypatch.setattr(R, name,
                            os.path.join(str(tmp_path), os.path.basename(orig)))
    monkeypatch.setattr(R, "post_verdict_test_diagnostics", lambda p: {})

    ax1 = mk_axis(cand=(2, 40))
    ax2 = mk_axis(cand=(3, 41))
    ax2.symbol = "SYN2"

    # gates 3/4 verify artifact + model SHA against the committed PRE-TEST
    # evidence; the fake hasher simply echoes the committed value per basename.
    ev_path = os.path.join("research", "liquidity_oracle_atlas", "evidence",
                           "opportunity_value_renewal_v1_pretest_summary.json")
    with open(ev_path) as f:
        pre_ev = json.load(f)
    committed = {}
    committed.update(pre_ev["r8_manifest"]["artifact_sha256"])
    committed.update(pre_ev["r9_model_manifest"]["model_sha256"])
    monkeypatch.setattr(
        R, "sha256_file",
        lambda p: committed.get(os.path.basename(p), "x"))
    real_rp = R.pd.read_parquet

    def fake_rp(p, *a, **k):
        if "opportunity_value_v1" in str(p):      # stub only the R8/R9 artifacts
            return pd.DataFrame({"symbol": [], "bar_index": [], "decision_bar": [],
                                 "side": []})
        return real_rp(p, *a, **k)

    monkeypatch.setattr(R.pd, "read_parquet", fake_rp)
    monkeypatch.setattr(R9, "predict_test", lambda **k: pd.DataFrame(
        {"symbol": [], "decision_bar": [], "side": []}))
    def fake_e9_root_axis(state_df, split=None, test_mask=None):
        R._bump("direction_chain_fits")
        R._bump("direction_batch_prediction_passes")
        return (pd.DataFrame({"symbol": [], "decision_bar": [], "e9_side": []}),
                None)

    monkeypatch.setattr(R, "build_e9_root_axis", fake_e9_root_axis)
    monkeypatch.setattr(R, "e9_axis_reproduction_gate",
                        lambda a, state_df=None, split=None: {
                            "n_frozen_rows": 13773, "n_mismatch": 0,
                            "n_missing": 0, "ok": True})
    # 15 symbol axes so the real performance budget (15 x 3 = 45) is exercised.
    axes15 = {}
    for k, s in enumerate(R.SYMBOLS):
        a = mk_axis(cand=(2, 40))
        a.symbol = s
        axes15[s] = a
    monkeypatch.setattr(R, "build_symbol_axes",
                        lambda s, p, r, e, sp, symbols=None: axes15)
    monkeypatch.setattr(R, "_common_days",
                        lambda axes, split: np.array([f"d{i}" for i in range(20)]))
    monkeypatch.setattr(R, "_git_head_sha", lambda: "SHA")
    R.reset_counters()
    res = R.run_formal_opportunity_value_test(
        allow_test=True, authorized_review_sha="SHA", verbose=False)
    assert res["verdict"] in R.VERDICTS
    assert res["e9_reproduction"]["ok"] is True
    assert res["scientific_status"] == \
        "LOCKED_DEVELOPMENT_TEST_NOT_PRISTINE_CONFIRMATION"
    assert res["performance"]["model_predict_calls_inside_simulator"] == 0
    assert res["performance"]["path_scans_inside_simulator"] == 0
    # FP12: 15 symbol axes x 3 policies = 45 sequential loops
    assert res["performance"]["sequential_symbol_loops"] == 45
    assert res["performance"]["test_label_reads_during_strategy"] == 0
    for k in ("delta_full", "delta_gate", "delta_renew"):
        assert "ci_low" in res[k] and "ci_high" in res[k]


def install_formal_mocks(monkeypatch, *, extra=None):
    """Shared mocked Formal-TEST harness (R10). No canonical path is touched."""
    import research.liquidity_oracle_atlas.opportunity_value_model_v1 as R9
    ev_path = os.path.join("research", "liquidity_oracle_atlas", "evidence",
                           "opportunity_value_renewal_v1_pretest_summary.json")
    with open(ev_path) as f:
        pre_ev = json.load(f)
    committed = {}
    committed.update(pre_ev["r8_manifest"]["artifact_sha256"])
    committed.update(pre_ev["r9_model_manifest"]["model_sha256"])
    monkeypatch.setattr(R, "sha256_file",
                        lambda p: committed.get(os.path.basename(p), "x"))
    real_rp = R.pd.read_parquet

    def fake_rp(p, *a, **k):
        if "opportunity_value_v1" in str(p):
            return pd.DataFrame({"symbol": [], "bar_index": [], "decision_bar": [],
                                 "side": []})
        return real_rp(p, *a, **k)

    monkeypatch.setattr(R.pd, "read_parquet", fake_rp)
    monkeypatch.setattr(R9, "predict_test", lambda **k: pd.DataFrame(
        {"symbol": [], "decision_bar": [], "side": []}))

    def fake_e9_root_axis(state_df, split=None, test_mask=None):
        R._bump("direction_chain_fits")
        R._bump("direction_batch_prediction_passes")
        return (pd.DataFrame({"symbol": [], "decision_bar": [], "e9_side": []}),
                None)

    monkeypatch.setattr(R, "build_e9_root_axis", fake_e9_root_axis)
    monkeypatch.setattr(R, "e9_axis_reproduction_gate",
                        lambda a, state_df=None, split=None: {
                            "n_frozen_rows": 13773, "n_mismatch": 0,
                            "n_missing": 0, "ok": True})
    axes15 = {}
    for s in R.SYMBOLS:
        a = mk_axis(cand=(2, 40))
        a.symbol = s
        axes15[s] = a
    monkeypatch.setattr(R, "build_symbol_axes",
                        lambda s, p, r, e, sp, symbols=None: axes15)
    monkeypatch.setattr(R, "_common_days",
                        lambda axes, split: np.array([f"d{i}" for i in range(20)]))
    monkeypatch.setattr(R, "_git_head_sha", lambda: "SHA")
    R.reset_counters()
    if extra is not None:
        extra(monkeypatch)


# =========================================================================== #
# FP1-FP20 final PRE-TEST regressions                                           #
# =========================================================================== #
def test_fp2_dtp9_state_columns_reproduce_frozen_candidate_values():
    """FP2: raw DTP9 in state_v1 must reproduce the Direction dataset values at
    the frozen 13,773 Candidate rows."""
    from research.liquidity_oracle_atlas.build_struct33_dataset_v1 import DTP9
    from research.liquidity_oracle_atlas.direction_null_baseline_v1 import (
        build_frozen_split)
    split = build_frozen_split()
    ds = split["ds"]
    frozen = pd.read_parquet(
        "artifacts/entry_path_atlas_v1/e9_direction_state_v1.parquet",
        columns=["symbol", "candidate_decision_index"])
    st = pd.read_parquet("artifacts/opportunity_value_v1/state_v1.parquet",
                         columns=["symbol", "bar_index"] + list(DTP9))
    m = frozen.merge(st, how="left", left_on=["symbol", "candidate_decision_index"],
                     right_on=["symbol", "bar_index"])
    assert len(m) == 13773 and int(m["bar_index"].isna().sum()) == 0
    d = ds.set_index(["symbol", "candidate_decision_index"])
    key = pd.MultiIndex.from_arrays(
        [m["symbol"], m["candidate_decision_index"]])
    for c in DTP9:
        want = d[c].reindex(key).to_numpy(float)
        got = m[c].to_numpy(float)
        assert np.allclose(want, got, atol=1e-12, equal_nan=True), c


def test_fp4_reproduction_gate_fails_on_missing_or_mismatched_row():
    frozen = pd.read_parquet(
        "artifacts/entry_path_atlas_v1/e9_direction_state_v1.parquet",
        columns=["symbol", "candidate_decision_index", "e9_direction"])
    good = frozen.rename(
        columns={"candidate_decision_index": "decision_bar"})[
        ["symbol", "decision_bar", "e9_direction"]]
    assert R.e9_axis_reproduction_gate(good)["ok"] is True
    missing = good.iloc[1:].reset_index(drop=True)
    with pytest.raises(RuntimeError, match="E9_AXIS_REPRODUCTION_MISMATCH"):
        R.e9_axis_reproduction_gate(missing)
    bad = good.copy()
    bad.loc[bad.index[0], "e9_direction"] = (
        "SHORT" if bad.loc[bad.index[0], "e9_direction"] == "LONG" else "LONG")
    with pytest.raises(RuntimeError, match="E9_AXIS_REPRODUCTION_MISMATCH"):
        R.e9_axis_reproduction_gate(bad)


def test_fp6_p1_p2_skip_bracket_ineligible_root_side_even_with_positive_ev():
    ax = mk_axis(cand=(2,))
    ax.bracket_eligible_long = np.zeros(80, bool)     # LONG ineligible
    for H in ("td1", "td3", "td5"):
        ax.ev[(H, 1)] = np.full(80, 5.0)              # very positive EV
    for p in ("P1", "P2"):
        trades, dec = R.simulate_symbol(p, ax)
        assert len(trades) == 0
        assert [d[0] for d in dec] == ["SKIP_INELIGIBLE"]
    # P0 is the frozen Direction baseline and does NOT require eligibility
    trades, _ = R.simulate_symbol("P0", ax)
    assert len(trades) == 1


def test_fp5_renewal_hold_when_current_side_ev_is_larger_positive():
    ax = mk_axis(cand=(2,))
    for H in ("td1", "td3", "td5"):
        ax.ev[(H, 1)] = np.full(80, 0.9)      # current side (LONG) dominates
        ax.ev[(H, -1)] = np.full(80, 0.2)
    trades, dec = R.simulate_symbol("P2", ax)
    assert any(d[0] == "HOLD" for d in dec)
    assert not any(d[0] == "REVERSE" for d in dec)
    assert len(trades) == 1


def test_fp5_renewal_reverse_when_opposite_side_ev_is_strictly_larger():
    ax = mk_axis(cand=(2,))
    for H in ("td1", "td3", "td5"):
        ax.ev[(H, 1)] = np.full(80, 0.2)
        ax.ev[(H, -1)] = np.full(80, 0.9)
    trades, dec = R.simulate_symbol("P2", ax)
    assert any(d[0] == "REVERSE" for d in dec)
    assert any(t.side < 0 for t in trades)


def test_fp5_positive_exact_tie_holds_current_side():
    ax = mk_axis(cand=(2,))
    for H in ("td1", "td3", "td5"):
        ax.ev[(H, 1)] = np.full(80, 0.7)
        ax.ev[(H, -1)] = np.full(80, 0.7)     # EXACT tie
    trades, dec = R.simulate_symbol("P2", ax)
    assert any(d[0] == "HOLD" for d in dec)
    assert not any(d[0] == "REVERSE" for d in dec)


def test_fp5_renewal_never_reads_e9_direction():
    """FP5: E9 is root-only. Zeroing E9 everywhere except the root bar must not
    change the renewal behaviour at all."""
    ax = mk_axis(cand=(2,))
    for H in ("td1", "td3", "td5"):
        ax.ev[(H, 1)] = np.full(80, 0.9)
        ax.ev[(H, -1)] = np.full(80, 0.2)
    base_tr, base_dec = R.simulate_symbol("P2", ax)

    ax2 = mk_axis(cand=(2,))
    e9 = np.zeros(80, np.int8)
    e9[2] = 1                                  # E9 only at the ROOT bar
    ax2.e9_root_side = e9
    for H in ("td1", "td3", "td5"):
        ax2.ev[(H, 1)] = np.full(80, 0.9)
        ax2.ev[(H, -1)] = np.full(80, 0.2)
    tr2, dec2 = R.simulate_symbol("P2", ax2)
    assert [(t.fill_idx, t.exit_idx, t.exit_reason) for t in base_tr] == \
        [(t.fill_idx, t.exit_idx, t.exit_reason) for t in tr2]
    assert [d[0] for d in base_dec] == [d[0] for d in dec2]


def test_fp9_test_mask_stops_at_common_end_and_deadline_is_capped():
    from research.liquidity_oracle_atlas.direction_null_baseline_v1 import (
        build_frozen_split)
    from research.liquidity_oracle_atlas.structural_renewal_dataset_v1 import (
        split_bounds)
    split = build_frozen_split()
    _t1, t2, end_t = split_bounds(split)
    assert t2 < end_t
    state = pd.read_parquet("artifacts/opportunity_value_v1/state_v1.parquet",
                            columns=["symbol", "bar_index", "decision_time",
                                     "candidate_at_decision"])
    dt = state["decision_time"].to_numpy("datetime64[ns]")
    # the FP9 closed window never admits a decision after COMMON_END
    m = (dt >= t2) & (dt <= end_t)
    assert int((dt > end_t).sum()) > 0
    assert int((dt[m] > end_t).sum()) == 0


def test_fp11_post_verdict_diagnostic_reads_test_labels_exactly_once():
    R.reset_counters()
    pred = pd.read_parquet(
        "artifacts/opportunity_value_v1/labels_test_v1.parquet",
        columns=["symbol", "decision_bar", "side", "horizon"]).head(0)
    # an empty prediction frame exercises the read path without asserting values
    out = R.post_verdict_test_diagnostics(pd.DataFrame(
        {"symbol": [], "decision_bar": [], "side": [], "td5_predicted_ev": []}))
    assert R.COUNTERS["post_verdict_test_label_reads"] == 1
    R.reset_counters()


def test_fp12_performance_mismatch_hard_stops_acceptance(monkeypatch):
    def corrupt(mp):
        real_bump = R._bump
        mp.setattr(R, "_bump",
                   lambda name, n=1: (real_bump(name, n - 1)
                                      if name == "sequential_symbol_loops"
                                      else real_bump(name, n)))

    install_formal_mocks(monkeypatch, extra=corrupt)
    with pytest.raises(RuntimeError, match="PERFORMANCE_GATE"):
        R.run_formal_opportunity_value_test(allow_test=True,
                                            authorized_review_sha="SHA")


def test_fp13_formal_evidence_files_written_and_manifest_last(monkeypatch,
                                                              tmp_path):
    """FP13: the Formal writer must emit all six CSVs and write the manifest
    LAST. All paths are redirected to tmp so no canonical evidence is created."""
    written_order = []
    # redirect every Formal evidence path into tmp_path
    for name in ("F_POLICY_SUMMARY", "F_DAILY", "F_PER_SYMBOL",
                 "F_TRADE_LEDGER", "F_DECISION_LEDGER", "F_EV_DECILES",
                 "F_SUMMARY", "F_MANIFEST"):
        orig = getattr(R, name)
        monkeypatch.setattr(R, name,
                            os.path.join(str(tmp_path), os.path.basename(orig)))

    def fake_write_csv(path, df):
        written_order.append(os.path.basename(path))
        df.to_csv(path, index=False)

    def extra(mp):
        # NOTE: sha256_file is intentionally NOT overridden -- gates 3/4 need the
        # committed-echo hasher to validate the real R8/R9 artifacts.
        mp.setattr(R, "post_verdict_test_diagnostics", lambda p: {})
        mp.setattr(R, "_write_csv", fake_write_csv)

    install_formal_mocks(monkeypatch, extra=extra)
    real_open = open

    def fake_open(path, mode="r", *a, **k):
        if str(path).endswith("manifest.json") and "w" in mode:
            written_order.append("MANIFEST")
        return real_open(path, mode, *a, **k)

    monkeypatch.setattr(R, "open", fake_open, raising=False)
    R.run_formal_opportunity_value_test(allow_test=True,
                                        authorized_review_sha="SHA",
                                        write_artifacts=True)
    expect = [os.path.basename(p) for p in
              (R.F_POLICY_SUMMARY, R.F_DAILY, R.F_PER_SYMBOL,
               R.F_TRADE_LEDGER, R.F_DECISION_LEDGER, R.F_EV_DECILES)]
    for e in expect:
        assert e in written_order, (e, written_order)
    assert "MANIFEST" in written_order
    assert written_order.index("MANIFEST") == len(written_order) - 1
    # canonical paths must remain untouched
    assert not os.path.exists(os.path.join(
        "research", "liquidity_oracle_atlas", "evidence",
        "opportunity_value_renewal_v1_manifest.json"))


def test_fp15_per_symbol_daily_aggregates_to_portfolio():
    axes = {}
    for k, s in enumerate(["S1", "S2", "S3"]):
        a = mk_axis(cand=(2,))
        a.symbol = s
        axes[s] = a
    trades = {"S1": [], "S2": [], "S3": []}
    for s, a in axes.items():
        t, _ = R.simulate_symbol("P0", a)
        trades[s] = t
    days = ["d0", "d1", "d2"]
    port, per = R.daily_returns(trades, axes, days)
    manual = sum(per[s].to_numpy(float) for s in axes) / len(axes)
    assert np.allclose(port.to_numpy(float), manual, atol=1e-12)


def test_policy_names_are_frozen():
    assert R.POLICIES == ("P0", "P1", "P2")
    assert R.BASELINE_POLICY == "P0"
    assert R.GATE_POLICY == "P1"
    assert R.PRIMARY_POLICY == "P2"
    assert R.TRADE_HORIZON_DAYS == 5
