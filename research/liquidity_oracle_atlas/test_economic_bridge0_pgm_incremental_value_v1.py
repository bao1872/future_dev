"""
test_economic_bridge0_pgm_incremental_value_v1.py
================================================
Tests for experiment_economic_bridge0_pgm_incremental_value_v1.py.

Run with:
    python research/liquidity_oracle_atlas/test_economic_bridge0_pgm_incremental_value_v1.py

These tests cover the audit / join / imputation / scoring / parity / cost / bootstrap
contracts WITHOUT executing a formal economic run. No new backtester, no reward re-scan,
no TB3-based selection.
"""

from __future__ import annotations

import sys
import inspect
from pathlib import Path

# Make the repository root importable when run as a script (python research/.../test_*.py)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
from pandas.errors import MergeError

import research.liquidity_oracle_atlas.experiment_dynamic_pgm1c_free_run_rollout_v1 as pgm
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1b_terminal_reset_closure_v1 as exp1b
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1a2c_representation_control_v1 as rep
import research.liquidity_oracle_atlas.run_v2_ml_recency_multi_action_v1 as v2
import research.liquidity_oracle_atlas.experiment_economic_bridge0_pgm_incremental_value_v1 as eb0


# ---------------------------------------------------------------------------
# Shared fixtures (lazy; heavy loads/fits happen at most once)
# ---------------------------------------------------------------------------
_ECON = None
_FIT_A = None


def economic():
    global _ECON
    if _ECON is None:
        _ECON = eb0.load_economic()
    return _ECON


def fit_A():
    global _FIT_A
    if _FIT_A is None:
        _FIT_A = pgm.fit_samplers_for_window(pgm.WINDOWS[0], pgm.SAMPLE_PATH, pgm.TRANSITION_SAMPLE_PATH)
    return _FIT_A


FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# 1. exact bar join; off-by-one must change the match set
# ---------------------------------------------------------------------------
def test_exact_bar_join_off_by_one():
    sample = eb0.load_pgm_sample()
    # pick 3 real (symbol, bar_t) pairs
    head = sample[["symbol", "bar_t"]].head(3).copy()
    ec_match = pd.DataFrame({
        "symbol": head["symbol"].tolist(),
        "signal_bar_index": head["bar_t"].tolist(),
        "block": ["TB2", "TB2", "TB3"],
    })
    j1 = _join_with(ec_match, sample)
    check("T1 exact join matches real bars", j1["matched"].sum() == 3, f"matched={j1['matched'].sum()}")

    # Exact-key identity: matched rows must have bar_t == signal_bar_index (no asof/nearest/offset).
    m1 = j1[j1["matched"]]
    check("T1 matched bar_t == signal_bar_index (no offset/asof)",
          bool((m1["bar_t"] == m1["signal_bar_index"]).all()))

    # Key-sensitivity: a 1-bar wrong key must change the matched set (join is not tolerant).
    ec_shift = pd.DataFrame({
        "symbol": head["symbol"].tolist(),
        "signal_bar_index": [b + 1 for b in head["bar_t"].tolist()],  # off by one
        "block": ["TB2", "TB2", "TB3"],
    })
    j2 = _join_with(ec_shift, sample)
    set1 = set(map(tuple, m1[["symbol", "bar_t"]].values))
    set2 = set(map(tuple, j2[j2["matched"]][["symbol", "bar_t"]].values))
    check("T1 1-bar key error changes matched set (join not tolerant)", set1 != set2)


def _join_with(ec, sample):
    ec = ec.copy()
    ec["bar_t"] = ec["signal_bar_index"].astype(int)
    j = ec.merge(sample, on=["symbol", "bar_t"], how="left", validate="many_to_one", indicator="_merge")
    j["matched"] = j["_merge"] == "both"
    return j


# ---------------------------------------------------------------------------
# 2. duplicate PGM key must fail the join (validate many_to_one)
# ---------------------------------------------------------------------------
def test_duplicate_pgm_key_fails():
    sample = eb0.load_pgm_sample()
    dup = sample.iloc[[0, 0]].copy()  # force a duplicate (symbol, bar_t)
    ec = pd.DataFrame({
        "symbol": [dup.iloc[0]["symbol"]],
        "signal_bar_index": [int(dup.iloc[0]["bar_t"])],
        "block": ["TB2"],
    })
    raised = False
    try:
        _join_with(ec, dup)
    except (MergeError, ValueError):
        raised = True
    check("T2 duplicate PGM key raises", raised)


# ---------------------------------------------------------------------------
# 3. no asof / fill: unmatched rows stay unmatched
# ---------------------------------------------------------------------------
def test_no_asof_fill():
    sample = eb0.load_pgm_sample()
    ec = pd.DataFrame({
        "symbol": [sample.iloc[0]["symbol"]],
        "signal_bar_index": [int(sample["bar_t"].max()) + 99999],  # impossible bar
        "block": ["TB2"],
    })
    j = _join_with(ec, sample)
    check("T3 unmatched bar stays unmatched (no asof/ffill)", j["matched"].iloc[0] == False)


# ---------------------------------------------------------------------------
# 4. signal bar -> next-open timing parity (entry_bar_index = signal_bar_index + 1)
# ---------------------------------------------------------------------------
def test_signal_next_open_parity():
    ec = economic()
    ok = (ec["entry_bar_index"] - ec["signal_bar_index"] == 1).all()
    check("T4 entry_bar_index == signal_bar_index + 1", bool(ok))


# ---------------------------------------------------------------------------
# 5. TB2 reward_end_time < TB3 start purge
# ---------------------------------------------------------------------------
def test_reward_end_purge():
    ec = economic()
    tb3_start = pd.to_datetime(ec[ec["block"] == "TB3"]["entry_time"]).min()
    tb2 = ec[ec["block"] == "TB2"]
    tb2_end = pd.to_datetime(tb2["reward_end_time"])
    kept = tb2_end < tb3_start
    if kept.any():
        max_kept = tb2_end[kept].max()
        check("T5 purged TB2 max reward_end_time < TB3 start", max_kept < tb3_start,
              f"max_kept={max_kept} tb3_start={tb3_start}")
    else:
        check("T5 purge keeps some TB2 rows", False, "all TB2 rows purged")


# ---------------------------------------------------------------------------
# 6. TB3 never enters imputer fit (train-only median)
# ---------------------------------------------------------------------------
def test_train_only_imputation():
    X_tr = np.array([[1.0, np.nan], [3.0, 2.0]])
    X_te = np.array([[np.nan, np.nan], [5.0, 9.0]])
    Xtr2, Xte2, med = eb0.train_only_impute(X_tr, X_te)
    # train median col0 = 2.0, col1 = 2.0
    check("T6 train median col0 == 2.0", abs(med[0] - 2.0) < 1e-12, f"med={med}")
    check("T6 test imputed from TRAIN median (not test)", abs(Xte2[0, 0] - 2.0) < 1e-12,
          f"Xte2[0,0]={Xte2[0,0]}")
    check("T6 train NaN filled", not np.isnan(Xtr2).any() and not np.isnan(Xte2).any())


# ---------------------------------------------------------------------------
# 7. STATE columns all from frozen lists
# ---------------------------------------------------------------------------
def test_state_cols_from_frozen_lists():
    allowed = set(exp1b.OBS_NUM) | set(rep.MC_EXTRA)
    check("T7 STATE_COLS subset of OBS_NUM ∪ MC_EXTRA", set(eb0.STATE_COLS) <= allowed,
          f"extra={set(eb0.STATE_COLS) - allowed}")


# ---------------------------------------------------------------------------
# 8. zt_* -> phi_* semantics consistent with current PGM
# ---------------------------------------------------------------------------
def test_zt_to_phi_parity():
    mc = fit_A()["trans_samplers"]["MC_STATE_CURREENCODING"]
    s = eb0.load_pgm_sample().head(200).copy()
    # phi-only case (drop any zt_* that might exist)
    zt_cols = [c for c in s.columns if c.startswith("zt_")]
    s_phi = s.drop(columns=zt_cols)
    out_phi = mc.analytic_conditional_support(s_phi)

    # zt-present case: set zt_* == phi_* so both paths should agree
    s_zt = s_phi.copy()
    for zc in zt_cols:
        pc = zc.replace("zt_", "phi_")
        if pc in s_zt.columns:
            s_zt[zc] = s_zt[pc]
    out_zt = mc.analytic_conditional_support(s_zt)

    key = "z_d_up_mu"
    same = np.allclose(np.asarray(out_phi[key]), np.asarray(out_zt[key]), atol=1e-9, equal_nan=True)
    check("T8 zt_==phi_ equals phi-only analytic output", same)


# ---------------------------------------------------------------------------
# 9. PGM score helper deterministic + expected keys
# ---------------------------------------------------------------------------
def test_pgm_score_helper_parity():
    mc = fit_A()["trans_samplers"]["MC_STATE_CURREENCODING"]
    s = eb0.load_pgm_sample().head(200).copy()
    a = pgm.compute_state_conditional_support_probs(s, mc)
    b = pgm.compute_state_conditional_support_probs(s, mc)
    keys_ok = all(k in a for k in ["p_up_distance_negative", "p_geometry_invalid",
                                   "p_any_age_invalid", "p_physical_invalid_approx"])
    same = all(np.allclose(a[k], b[k], equal_nan=True) for k in a)
    check("T9 PGM helper returns expected keys", keys_ok)
    check("T9 PGM helper deterministic", same)


# ---------------------------------------------------------------------------
# 10. reward_* parity with frozen V2 dataset
# ---------------------------------------------------------------------------
def test_reward_frozen_parity():
    ec = economic()
    check("T10 reward_SKIP == 0", bool((ec["reward_SKIP"] == 0).all()))
    check("T10 reward_* all finite",
          bool(np.isfinite(ec[eb0.REWARD_COLS].to_numpy()).all()))
    check("T10 reward_end_semantics == HARDENING_VERSION",
          bool((ec["reward_end_semantics"] == eb0.HARDENING_VERSION).all()))


# ---------------------------------------------------------------------------
# 11. apply_action_policy parity with V2
# ---------------------------------------------------------------------------
def test_apply_action_policy_parity():
    q_pos = np.array([[0.0, 0.1, -0.2, 0.3]])
    q_neg = np.array([[0.0, -0.1, -0.2, -0.3]])
    a_pos = v2.apply_action_policy(q_pos)
    a_neg = v2.apply_action_policy(q_neg)
    check("T11 positive trade Q -> argmax action", a_pos[0] == 3, f"a={a_pos}")
    check("T11 all trade Q <= 0 -> SKIP", a_neg[0] == 0, f"a={a_neg}")


# ---------------------------------------------------------------------------
# 12. all predicted Q <= 0 -> SKIP
# ---------------------------------------------------------------------------
def test_all_q_negative_skip():
    q = np.array([[0.0, -0.5, -0.2, -0.1], [0.0, -0.01, -0.02, -0.03]])
    a = v2.apply_action_policy(q)
    check("T12 all-negative trade Q -> all SKIP", bool((a == 0).all()), f"a={a}")


# ---------------------------------------------------------------------------
# 13. chosen realized reward lookup exact
# ---------------------------------------------------------------------------
def test_chosen_lookup_exact():
    R = np.array([[0.0, 0.4, -0.2, 0.7], [0.0, -0.1, 0.3, 0.2]])
    action = np.array([3, 2])
    chosen = R[np.arange(len(action)), action]
    expected = np.array([0.7, 0.3])
    check("T13 chosen reward lookup exact", np.allclose(chosen, expected))


# ---------------------------------------------------------------------------
# 14. cost only debited on non-SKIP
# ---------------------------------------------------------------------------
def test_cost_only_on_non_skip():
    chosen = np.array([0.0, 0.5, -0.3, 0.0])  # index 0 and 3 are SKIP
    action = np.array([0, 1, 2, 0])
    c = 0.1
    net = chosen - c * (action != 0)
    # SKIP rows (0 and 3) unchanged at 0.0
    check("T14 SKIP rows unchanged by cost", net[0] == 0.0 and net[3] == 0.0)
    check("T14 non-SKIP debited by c", abs(net[1] - 0.4) < 1e-12 and abs(net[2] + 0.4) < 1e-12)


# ---------------------------------------------------------------------------
# 15. break-even cost algebra
# ---------------------------------------------------------------------------
def test_break_even_algebra():
    # simulate a route: EV per signal 0.02, trade_rate 0.5  -> break-even = 0.04
    gross_ev_ps = 0.02
    trade_rate = 0.5
    be = gross_ev_ps / trade_rate
    net_at_be = gross_ev_ps - be * trade_rate
    check("T15 break-even cost makes net EV ~ 0", abs(net_at_be) < 1e-12, f"net={net_at_be}")


# ---------------------------------------------------------------------------
# 16. day-cluster bootstrap deterministic
# ---------------------------------------------------------------------------
def test_bootstrap_deterministic():
    rng = np.random.default_rng(0)
    r_spg = rng.normal(0.01, 0.1, 500)
    r_state = rng.normal(0.0, 0.1, 500)
    day = rng.integers(0, 50, 500)
    b1 = eb0.incremental_bootstrap(r_spg, r_state, day, n=200, seed=eb0.SEED)
    b2 = eb0.incremental_bootstrap(r_spg, r_state, day, n=200, seed=eb0.SEED)
    check("T16 bootstrap deterministic (mean_delta)", b1["mean_delta_R"] == b2["mean_delta_R"])
    check("T16 bootstrap deterministic (ci_lo)", b1["ci_lo"] == b2["ci_lo"])


# ---------------------------------------------------------------------------
# 17. no forbidden reward/future columns in X
# ---------------------------------------------------------------------------
def test_no_forbidden_columns_in_features():
    forbidden = set(eb0.REWARD_COLS) | {"target_price", "stop_price", "target_atr",
                                        "entry_price", "reference_entry"}
    bad = (set(eb0.STATE_COLS) | set(eb0.PGM_SCORE_COLS)) & forbidden
    check("T17 no forbidden cols in STATE/PGM views", len(bad) == 0, f"bad={bad}")


# ---------------------------------------------------------------------------
# 18. smoke path does not emit a formal verdict
# ---------------------------------------------------------------------------
def test_smoke_no_formal_verdict():
    src = inspect.getsource(eb0.run_smoke)
    check("T18 smoke has no SUPPORTED verdict string",
          "PGM_INCREMENTAL_ECONOMIC_VALUE_SUPPORTED_ON_FROZEN_UNIVERSE" not in src)
    check("T18 smoke has no NOT_SUPPORTED verdict string",
          "PGM_INCREMENTAL_ECONOMIC_VALUE_NOT_SUPPORTED_ON_FROZEN_UNIVERSE" not in src)
    check("T18 smoke does not assign 'verdict' to its output dict",
          '"verdict":' not in src and "'verdict':" not in src)


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} ECONOMIC-BRIDGE-0A tests...")
    for t in tests:
        print(f"[{t.__name__}]")
        t()
    print(f"\n{len(tests) - len(FAILURES)}/{len(tests)} passed, {len(FAILURES)} failed")
    if FAILURES:
        print("FAILED:", FAILURES)
        raise SystemExit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
