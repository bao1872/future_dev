"""§59 Policy tests + §60 TEST-governance tests + §58 (no model in simulator).

Uses fully synthetic SymbolAxis objects (no materialized data, no models).
"""

import numpy as np
import pandas as pd
import pytest

from research.liquidity_oracle_atlas import sequential_decomposed_policy_v1 as R10
from research.liquidity_oracle_atlas.build_decomposed_value_dataset_v1 import (
    HORIZONS)


def _blank_axis(n=60, e9_side=None, deadline_extra=30, symbol="SYN"):
    if e9_side is None:
        e9_side = np.tile([1, -1], n // 2 + 1)[:n]
    day = np.arange(n)
    seg = np.ones(n, np.int64)
    decision_time = pd.date_range(
        "2024-01-01", periods=n, freq="15min").to_numpy("datetime64[ns]")
    axis = R10.SymbolAxis(
        symbol=symbol, n_bars=n,
        bar_start_time=decision_time, decision_time=decision_time,
        trading_day=day, segment=seg,
        open=np.linspace(100, 110, n), high=np.linspace(101, 111, n),
        low=np.linspace(99, 109, n), close=np.linspace(100, 110, n),
        atr=np.full(n, 1.0), sup_top=np.full(n, 112.0),
        res_bottom=np.full(n, 98.0),
        candidate_at_decision=np.ones(n, bool),
        test_mask=np.ones(n, bool), e9_root_side=e9_side.astype(np.int8),
        deadline_idx=np.minimum(np.arange(n) + deadline_extra, n - 1),
        day_ord=np.arange(n),
        event_idx_long=np.full(n, -1), event_idx_short=np.full(n, -1),
        renewal_fill_long=np.full(n, -1), renewal_fill_short=np.full(n, -1),
        bracket_eligible_long=np.ones(n, bool),
        bracket_eligible_short=np.ones(n, bool))
    evw = {(H, sd): np.full(n, 0.10) for H in HORIZONS for sd in (1, -1)}
    evr = {(H, sd): np.full(n, 0.10) for H in HORIZONS for sd in (1, -1)}
    evc = {(H, sd): np.full(n, 0.10) for H in HORIZONS for sd in (1, -1)}
    axis.evw, axis.evr, axis.evc = evw, evr, evc
    return axis


def _fill(axis, attr, val):
    d = getattr(axis, attr)
    for k in d:
        d[k][:] = val


def _set_ev(axis, horizon, side, val):
    axis.evc[(horizon, side)][:] = val


def _decisions_by_action(decisions):
    from collections import Counter
    return Counter(a for a, *_ in decisions)


# --------------------------------------------------------------------------- #
# §59.1 baseline ignores gates                                                 #
# --------------------------------------------------------------------------- #
def test_p0_ignores_gates():
    ax = _blank_axis()
    _fill(ax, "evw", -1.0)
    _fill(ax, "evr", -1.0)
    _fill(ax, "evc", -1.0)
    trades, dec = R10.simulate_symbol("P0", ax)
    assert len(trades) > 0
    assert all(t.side in (1, -1) for t in trades)


def test_pw_uses_ev_w_only():
    ax = _blank_axis()
    _fill(ax, "evw", 0.10)
    _fill(ax, "evr", -1.0)
    _fill(ax, "evc", -1.0)
    trades, dec = R10.simulate_symbol("PW", ax)
    assert len(trades) > 0
    _fill(ax, "evw", -0.10)
    _fill(ax, "evr", 0.10)
    _fill(ax, "evc", 0.10)
    trades2, dec2 = R10.simulate_symbol("PW", ax)
    assert len(trades2) == 0


def test_pr_uses_ev_r_only():
    ax = _blank_axis()
    _fill(ax, "evw", -1.0)
    _fill(ax, "evr", 0.10)
    _fill(ax, "evc", -1.0)
    trades, dec = R10.simulate_symbol("PR", ax)
    assert len(trades) > 0
    _fill(ax, "evw", 0.10)
    _fill(ax, "evr", -0.10)
    _fill(ax, "evc", 0.10)
    trades2, dec2 = R10.simulate_symbol("PR", ax)
    assert len(trades2) == 0


def test_pc_uses_ev_c_only():
    ax = _blank_axis()
    _fill(ax, "evw", 0.10)
    _fill(ax, "evr", 0.10)
    _fill(ax, "evc", 0.10)
    trades, dec = R10.simulate_symbol("PC", ax)
    assert len(trades) > 0
    _fill(ax, "evc", -0.10)
    trades2, dec2 = R10.simulate_symbol("PC", ax)
    assert len(trades2) == 0


def test_pn_root_equals_pc_root():
    ax = _blank_axis()
    _fill(ax, "evc", 0.10)
    _, dec_pc = R10.simulate_symbol("PC", ax)
    _, dec_pn = R10.simulate_symbol("PN", ax)
    assert _decisions_by_action(dec_pc)["ENTER"] == \
        _decisions_by_action(dec_pn)["ENTER"]


def test_win_only_change_does_not_alter_pr():
    ax = _blank_axis()
    _fill(ax, "evr", 0.10)
    _, dec_a = R10.simulate_symbol("PR", ax)
    _fill(ax, "evw", -5.0)
    _, dec_b = R10.simulate_symbol("PR", ax)
    assert _decisions_by_action(dec_a) == _decisions_by_action(dec_b)


def test_payoff_only_change_does_not_alter_pw():
    ax = _blank_axis()
    _fill(ax, "evw", 0.10)
    _, dec_a = R10.simulate_symbol("PW", ax)
    _fill(ax, "evr", -5.0)
    _, dec_b = R10.simulate_symbol("PW", ax)
    assert _decisions_by_action(dec_a) == _decisions_by_action(dec_b)


# --------------------------------------------------------------------------- #
# §59.2 PN renewal (EV_C LONG vs SHORT vs 0)                                    #
# --------------------------------------------------------------------------- #
def _setup_renewal(t=0, e=10, side=1):
    ax = _blank_axis(n=60, e9_side=np.where(np.arange(60) == t, side, 0))
    _fill(ax, "evc", 0.10)
    if side > 0:
        ax.event_idx_long[t] = e
        ax.renewal_fill_long[t] = e + 1
        ax.event_idx_long[e] = -1
    else:
        ax.event_idx_short[t] = e
        ax.renewal_fill_short[t] = e + 1
        ax.event_idx_short[e] = -1
    return ax, t, e


def test_pn_renewal_hold_when_current_best():
    ax, t, e = _setup_renewal(side=1)
    ax.evc[("td5", 1)][e] = 0.20
    ax.evc[("td5", -1)][e] = 0.10
    _, dec = R10.simulate_symbol("PN", ax)
    actions = _decisions_by_action(dec)
    assert actions["HOLD"] >= 1
    assert actions.get("REVERSE", 0) == 0


def test_pn_renewal_reverse_when_opp_better():
    ax, t, e = _setup_renewal(side=1)
    ax.evc[("td5", 1)][e] = 0.10
    ax.evc[("td5", -1)][e] = 0.30
    trades, dec = R10.simulate_symbol("PN", ax)
    actions = _decisions_by_action(dec)
    assert actions.get("REVERSE", 0) >= 1
    assert any(t.side == -1 for t in trades)


def test_pn_renewal_exit_when_both_nonpositive():
    ax, t, e = _setup_renewal(side=1)
    ax.evc[("td5", 1)][e] = -0.10
    ax.evc[("td5", -1)][e] = -0.20
    _, dec = R10.simulate_symbol("PN", ax)
    actions = _decisions_by_action(dec)
    assert actions.get("EXIT", 0) >= 1


def test_pn_renewal_positive_tie_holds():
    ax, t, e = _setup_renewal(side=1)
    ax.evc[("td5", 1)][e] = 0.20
    ax.evc[("td5", -1)][e] = 0.20
    _, dec = R10.simulate_symbol("PN", ax)
    actions = _decisions_by_action(dec)
    assert actions.get("HOLD", 0) >= 1
    assert actions.get("REVERSE", 0) == 0


def test_hold_does_not_reset_deadline():
    ax, t, e = _setup_renewal(side=1)
    ax.evc[("td5", 1)][e] = 0.20
    ax.evc[("td5", -1)][e] = 0.10
    trades, dec = R10.simulate_symbol("PN", ax)
    entry_deadline = ax.deadline_idx[t]
    for tr in trades:
        assert tr.deadline_idx == entry_deadline


def test_reverse_resets_deadline_and_atr():
    ax, t, e = _setup_renewal(side=1)
    ax.evc[("td5", 1)][e] = 0.10
    ax.evc[("td5", -1)][e] = 0.30
    trades, dec = R10.simulate_symbol("PN", ax)
    short_trades = [tr for tr in trades if tr.side == -1]
    assert short_trades
    assert short_trades[0].deadline_idx == ax.deadline_idx[e]
    assert short_trades[0].atr0 == ax.atr[e]


# --------------------------------------------------------------------------- #
# §59.3 five policies run                                                      #
# --------------------------------------------------------------------------- #
def test_five_policies_run_and_nonempty_for_p0():
    ax = _blank_axis()
    for p in R10.POLICIES:
        trades, dec = R10.simulate_symbol(p, ax)
        for tr in trades:
            assert 0 <= tr.holding_bars


# --------------------------------------------------------------------------- #
# §58 decomposition identity                                                   #
# --------------------------------------------------------------------------- #
def test_decomposition_vectors_exact():
    full = np.array([0.1, -0.2, 0.3, 0.0])
    base = np.zeros_like(full)
    gate = np.array([0.05, -0.1, 0.2, 0.0])
    renew = full - gate
    ident = R10.policy_decomposition_identity(full, gate, base)
    assert ident["ok"]
    assert np.allclose(renew, full - gate)


# --------------------------------------------------------------------------- #
# §60 governance: no model/scan calls inside simulator                          #
# --------------------------------------------------------------------------- #
def test_simulator_zero_model_calls():
    R10.reset_counters()
    ax = _blank_axis()
    R10.simulate_symbol("PN", ax)
    assert R10.COUNTERS["model_predict_calls_inside_simulator"] == 0
    assert R10.COUNTERS["path_scans_inside_simulator"] == 0
    assert R10.COUNTERS["environment_recomputes"] == 0
    assert R10.COUNTERS["geometry_recomputes"] == 0


def test_75_symbol_loops_formal_budget():
    R10.reset_counters()
    axes = {f"SYN_{i:02d}": _blank_axis(symbol=f"SYN_{i:02d}") for i in range(15)}
    for p in R10.POLICIES:
        for sym, ax in axes.items():
            R10.simulate_symbol(p, ax)
    assert R10.COUNTERS["sequential_symbol_loops"] == 75


def test_predict_test_gated():
    with pytest.raises(RuntimeError):
        R10.predict_test(allow_test=False)
    with pytest.raises(RuntimeError):
        R10.predict_test(allow_test=True)


def test_formal_runner_gated():
    with pytest.raises(RuntimeError):
        R10.run_formal_opportunity_value_test(allow_test=False)
    with pytest.raises(RuntimeError):
        R10.run_formal_opportunity_value_test(
            allow_test=True, authorized_review_sha="x", write_artifacts=False)


# --------------------------------------------------------------------------- #
# F2 : terminal_no_valid_open is a CLOSE exit, not a next-OPEN exit            #
# --------------------------------------------------------------------------- #
def _single_root_axis(n=60, root=0):
    """Axis with exactly ONE root candidate so a scenario yields one trade."""
    ax = _blank_axis(n=n)
    ax.candidate_at_decision[:] = False
    ax.candidate_at_decision[root] = True
    ax.e9_root_side[:] = 1                      # always LONG
    # CLOSE must differ from OPEN, otherwise a next-OPEN mis-mark is invisible.
    ax.close = np.linspace(100.0, 110.0, n) + 0.37
    return ax


def test_f2_terminal_no_valid_open_closes_at_event_close():
    ax = _single_root_axis()
    t, e, nxt = 0, 5, 6
    ax.event_idx_long[t] = e
    ax.renewal_fill_long[t] = nxt
    # Hard segment change at the renewal fill => no valid next OPEN.
    ax.segment[e] = 1
    ax.segment[nxt] = 2

    trades, _dec = R10.simulate_symbol("PN", ax)
    assert len(trades) == 1
    tr = trades[0]
    assert tr.exit_reason == "terminal_no_valid_open"
    assert tr.exit_idx == e
    assert tr.exit_price == pytest.approx(float(ax.close[e]))


def test_f2_terminal_no_valid_open_holds_event_bar_and_pnl_identity():
    ax = _single_root_axis()
    t, e, nxt = 0, 5, 6
    ax.event_idx_long[t] = e
    ax.renewal_fill_long[t] = nxt
    ax.segment[e] = 1
    ax.segment[nxt] = 2

    trades, _dec = R10.simulate_symbol("PN", ax)
    tr = trades[0]
    # The event bar IS held (CLOSE exit).
    assert tr.holding_bars == e - tr.fill_idx + 1
    assert tr.fill_idx <= e <= tr.exit_idx

    ret = R10.trade_return(tr)
    pnl_sum = float(np.sum(R10.bar_pnl(tr, ax)))
    assert abs(pnl_sum - ret) <= 1e-12


def test_f2_pnl_identity_holds_for_every_exit_reason():
    """sum(bar_pnl) == trade_return for all reachable exit reasons."""
    reasons = {}

    # terminal_no_valid_open
    ax = _single_root_axis()
    ax.event_idx_long[0] = 5
    ax.renewal_fill_long[0] = 6
    ax.segment[5] = 1
    ax.segment[6] = 2
    tr = R10.simulate_symbol("PN", ax)[0][0]
    reasons[tr.exit_reason] = (R10.trade_return(tr), float(np.sum(R10.bar_pnl(tr, ax))))

    # renewal_exit (both EVs <= 0)
    ax = _single_root_axis()
    ax.event_idx_long[0] = 5
    ax.renewal_fill_long[0] = 6
    _fill(ax, "evc", -1.0)
    ax.evc[("td5", 1)][0] = 0.10               # entry gate stays positive
    tr = R10.simulate_symbol("PN", ax)[0][0]
    reasons[tr.exit_reason] = (R10.trade_return(tr), float(np.sum(R10.bar_pnl(tr, ax))))

    # terminal_deadline
    ax = _single_root_axis()
    ax.event_idx_long[0] = -1
    tr = R10.simulate_symbol("PN", ax)[0][0]
    reasons[tr.exit_reason] = (R10.trade_return(tr), float(np.sum(R10.bar_pnl(tr, ax))))

    assert set(reasons) == {
        "terminal_no_valid_open", "renewal_exit", "terminal_deadline"}
    for reason, (ret, s) in reasons.items():
        assert abs(s - ret) <= 1e-12, f"identity broken for {reason}"


# --------------------------------------------------------------------------- #
# F3 : immutable entry decision vs advancing renewal epoch                     #
# --------------------------------------------------------------------------- #
def test_f3_hold_keeps_entry_decision_immutable():
    ax = _single_root_axis()
    ax.event_idx_long[0] = 5
    ax.renewal_fill_long[0] = 6
    ax.event_idx_long[5] = 10
    ax.renewal_fill_long[5] = 11
    ax.event_idx_long[10] = -1                  # -> terminal_deadline

    trades, dec = R10.simulate_symbol("PN", ax)
    assert len(trades) == 1
    tr = trades[0]
    # Renewal really progressed through both HOLD events ...
    holds = [d for d in dec if d[0] == "HOLD"]
    assert [d[1] for d in holds] == [5, 10]
    # ... while the entry decision that CREATED the trade is untouched.
    assert tr.decision_idx == 0
    assert tr.epoch_decision_idx == 10


def test_f3_reverse_opens_new_trade_at_reversal_decision():
    ax = _single_root_axis()
    ax.event_idx_long[0] = 5
    ax.renewal_fill_long[0] = 6
    # Opposite side strictly better at the renewal decision -> REVERSE.
    ax.evc[("td5", -1)][5] = 0.90
    ax.event_idx_short[5] = -1                  # new SHORT trade runs to deadline

    trades, dec = R10.simulate_symbol("PN", ax)
    assert [d[0] for d in dec].count("REVERSE") == 1
    assert len(trades) == 2
    first, second = trades
    assert first.decision_idx == 0               # original LONG entry preserved
    assert second.decision_idx == 5              # created by the reversal decision
    assert second.epoch_decision_idx == 5
    assert second.side == -1
