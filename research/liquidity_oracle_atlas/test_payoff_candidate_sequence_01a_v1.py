import hashlib
import numpy as np
import pandas as pd
import pytest

import research.liquidity_oracle_atlas.payoff_candidate_sequence_01a_v1 as M
from research.liquidity_oracle_atlas.payoff_candidate_sequence_01a_v1 import (
    PAY8_COLS, STATIC, ARMS, build_candidate_event_history,
    build_candidate_ledger, meta_fold_splits, residual_targets, run_sequence_oof,
    analyze, run_t1, reset_counters,
)

PAY8_DUMMY = {c: 1.0 for c in PAY8_COLS}


def mk(symbol, side, decision_bar, decision_time, label_available_time,
       episode_return_atr, p_win, mu_win, mu_loss, fold=None, **extra):
    row = dict(
        symbol=symbol, side=side, decision_bar=int(decision_bar),
        decision_time=pd.Timestamp(decision_time),
        label_available_time=pd.Timestamp(label_available_time),
        episode_return_atr=float(episode_return_atr),
        p_win=float(p_win), mu_win=float(mu_win), mu_loss=float(mu_loss),
    )
    row.update(PAY8_DUMMY)
    if fold is not None:
        row["fold"] = fold
    row.update(extra)
    return row


def make_ledger(rows):
    df = pd.DataFrame(rows)
    for c in PAY8_COLS:
        df[c] = df[c].astype(float)
    return df


# --------------------------------------------------------------------------- #
# A. Previous-event ordering (irregular bar gaps)                               #
# --------------------------------------------------------------------------- #
def test_previous_event_ordering_not_bar_lag():
    rows = [
        mk("AG", "LONG", 10, "2024-01-01 09:00", "2024-01-02 09:00",
           0.5, 0.50, 1.0, 0.5),
        mk("AG", "LONG", 25, "2024-01-01 10:00", "2024-01-02 10:00",
           -0.5, 0.60, 1.1, 0.6),
        mk("AG", "LONG", 40, "2024-01-01 11:00", "2024-01-02 11:00",
           1.0, 0.70, 1.2, 0.7),
        mk("AG", "LONG", 100, "2024-01-01 12:00", "2024-01-02 12:00",
           2.0, 0.80, 1.3, 0.8),
    ]
    df = make_ledger(rows)
    out = build_candidate_event_history(df)
    cur = out.iloc[-1]  # decision_bar = 100
    # lag1 = bar 40, lag2 = bar 25, lag3 = bar 10 (previous CANDIDATE events)
    assert cur["h1__gap_bars"] == 100 - 40
    assert cur["h2__gap_bars"] == 100 - 25
    assert cur["h3__gap_bars"] == 100 - 10
    assert cur["h1__p_win"] == 0.70   # bar 40
    assert cur["h2__p_win"] == 0.60   # bar 25
    assert cur["h3__p_win"] == 0.50   # bar 10
    # NOT adjacent bars: 100-99, 99-98, ...
    assert cur["h1__gap_bars"] != 1


# --------------------------------------------------------------------------- #
# B. Same-side isolation                                                       #
# --------------------------------------------------------------------------- #
def test_same_side_isolation():
    rows = []
    # LONG group
    rows.append(mk("AG", "LONG", 10, "2024-01-01 09:00", "2024-01-02 09:00",
                   0.5, 0.50, 1.0, 0.5))
    rows.append(mk("AG", "LONG", 20, "2024-01-01 10:00", "2024-01-02 10:00",
                   0.5, 0.55, 1.1, 0.5))
    rows.append(mk("AG", "LONG", 30, "2024-01-01 11:00", "2024-01-02 11:00",
                   0.5, 0.60, 1.2, 0.5))
    # SHORT group (distinct p_win stream)
    rows.append(mk("AG", "SHORT", 10, "2024-01-01 09:00", "2024-01-02 09:00",
                   0.5, 0.10, 1.0, 0.5))
    rows.append(mk("AG", "SHORT", 20, "2024-01-01 10:00", "2024-01-02 10:00",
                   0.5, 0.15, 1.1, 0.5))
    rows.append(mk("AG", "SHORT", 30, "2024-01-01 11:00", "2024-01-02 11:00",
                   0.5, 0.20, 1.2, 0.5))
    df = make_ledger(rows)
    out = build_candidate_event_history(df)
    long_cur = out[(out["symbol"] == "AG") & (out["side"] == "LONG")].iloc[-1]
    short_cur = out[(out["symbol"] == "AG") & (out["side"] == "SHORT")].iloc[-1]
    # LONG history lag1 must be the previous LONG p_win (0.55), not any SHORT
    assert long_cur["h1__p_win"] == 0.55
    assert short_cur["h1__p_win"] == 0.15
    assert long_cur["h1__p_win"] != short_cur["h1__p_win"]


# --------------------------------------------------------------------------- #
# C. Unresolved event stays in its slot                                        #
# --------------------------------------------------------------------------- #
def test_unresolved_event_keeps_slot():
    rows = [
        # C1 (earliest, but resolved before current)
        mk("AG", "LONG", 10, "2024-01-01 09:00", "2024-01-02 09:00",
           1.0, 0.40, 1.0, 0.5),
        # C2 (most recent previous; label available AFTER current -> unresolved)
        mk("AG", "LONG", 20, "2024-01-03 09:00", "2024-01-20 09:00",
           1.0, 0.45, 1.1, 0.5),
        # current
        mk("AG", "LONG", 30, "2024-01-05 09:00", "2024-01-06 09:00",
           1.0, 0.50, 1.2, 0.5),
    ]
    df = make_ledger(rows)
    out = build_candidate_event_history(df)
    cur = out.iloc[-1]
    # h1 = C2 (most recent previous) -> unresolved
    assert cur["h1__resolved"] == 0
    # h2 = C1 -> resolved
    assert cur["h2__resolved"] == 1
    # do NOT promote C1 to lag1
    assert cur["h1__p_win"] == 0.45   # C2
    assert cur["h2__p_win"] == 0.40   # C1
    assert cur["h1__return"] != cur["h1__return"] or pd.isna(cur["h1__return"])


# --------------------------------------------------------------------------- #
# D. Label-availability gate                                                   #
# --------------------------------------------------------------------------- #
def test_label_availability_gate():
    rows = [
        mk("AG", "LONG", 10, "2024-01-01 09:00", "2024-01-10 09:00",
           3.0, 0.25, 1.0, 0.8),   # prev resolved only if avail < current dt
        mk("AG", "LONG", 20, "2024-01-05 09:00", "2024-01-06 09:00",
           1.0, 0.50, 1.0, 0.5),
    ]
    df = make_ledger(rows)
    out = build_candidate_event_history(df)
    cur = out.iloc[-1]
    # prev label_available_time (2024-01-10) >= current decision_time (2024-01-05)
    assert cur["h1__resolved"] == 0
    assert pd.isna(cur["h1__return"])
    assert pd.isna(cur["h1__win"])
    assert pd.isna(cur["h1__prob_surprise"])
    assert pd.isna(cur["h1__econ_surprise"])


# --------------------------------------------------------------------------- #
# E. Availability mutation                                                     #
# --------------------------------------------------------------------------- #
def test_availability_mutation_does_not_change_features():
    rows = [
        mk("AG", "LONG", 10, "2024-01-01 09:00", "2024-01-20 09:00",
           3.0, 0.25, 1.0, 0.8),
        mk("AG", "LONG", 20, "2024-01-05 09:00", "2024-01-06 09:00",
           1.0, 0.50, 1.0, 0.5),
    ]
    df = make_ledger(rows)
    before = build_candidate_event_history(df).iloc[-1]
    # mutate the historical outcome that is NOT yet available at current time
    mut = df.copy()
    mut.loc[mut.index[0], "episode_return_atr"] = -9.0  # prev event
    after = build_candidate_event_history(mut).iloc[-1]
    assert before["h1__resolved"] == 0
    assert after["h1__resolved"] == 0
    assert pd.isna(before["h1__return"]) and pd.isna(after["h1__return"])
    assert before["h1__p_win"] == after["h1__p_win"]


# --------------------------------------------------------------------------- #
# F. Future mutation                                                           #
# --------------------------------------------------------------------------- #
def test_future_mutation_does_not_change_current_history():
    # current = bar 30; its history references bars 20 (h1) and 10 (h2).
    # A FUTURE event (bar 40) must never appear in current's past, so mutating
    # it cannot change current's history features.
    rows = [
        mk("AG", "LONG", 10, "2024-01-01 09:00", "2024-01-02 09:00",
           1.0, 0.40, 1.0, 0.5),
        mk("AG", "LONG", 20, "2024-01-03 09:00", "2024-01-04 09:00",
           1.0, 0.45, 1.1, 0.5),
        mk("AG", "LONG", 30, "2024-01-05 09:00", "2024-01-06 09:00",
           1.0, 0.50, 1.2, 0.5),
        mk("AG", "LONG", 40, "2024-01-07 09:00", "2024-01-08 09:00",
           1.0, 0.60, 1.3, 0.5),
    ]
    df = make_ledger(rows)
    b = build_candidate_event_history(df).iloc[2]  # current = bar 30
    mut = df.copy()
    mut.loc[mut.index[3], "p_win"] = 0.01  # future event (bar 40) mutated
    mut.loc[mut.index[3], "episode_return_atr"] = -9.0
    a = build_candidate_event_history(mut).iloc[2]
    # current history references only bars 20 and 10; unchanged.
    assert b["h1__p_win"] == a["h1__p_win"] == 0.45
    assert b["h2__p_win"] == a["h2__p_win"] == 0.40
    assert b["h1__gap_bars"] == a["h1__gap_bars"] == 10
    assert b["h2__gap_bars"] == a["h2__gap_bars"] == 20


# --------------------------------------------------------------------------- #
# G. Historical prediction missing                                             #
# --------------------------------------------------------------------------- #
def test_historical_prediction_missing():
    rows = [
        mk("AG", "LONG", 10, "2024-01-01 09:00", "2024-01-02 09:00",
           1.0, np.nan, np.nan, np.nan),  # no frozen OOF prediction
        mk("AG", "LONG", 20, "2024-01-03 09:00", "2024-01-04 09:00",
           1.0, 0.50, 1.0, 0.5),
    ]
    df = make_ledger(rows)
    out = build_candidate_event_history(df)
    cur = out.iloc[-1]
    assert cur["h1__exists"] == 1
    assert cur["h1__pred_available"] == 0
    assert pd.isna(cur["h1__p_win"])
    assert pd.isna(cur["h1__delta_p"])
    assert pd.isna(cur["h1__mu_win"])


# --------------------------------------------------------------------------- #
# H. Surprise hand calculation                                                 #
# --------------------------------------------------------------------------- #
def test_surprise_hand_calculation():
    rows = [
        mk("AG", "LONG", 10, "2024-01-01 09:00", "2024-01-02 09:00",
           3.0, 0.25, 1.0, 0.8),   # prev: winner, p=0.25, muW=1.0
        mk("AG", "LONG", 20, "2024-01-03 09:00", "2024-01-04 09:00",
           1.0, 0.50, 1.0, 0.5),   # current
    ]
    df = make_ledger(rows)
    out = build_candidate_event_history(df)
    cur = out.iloc[-1]
    assert cur["h1__resolved"] == 1
    assert cur["h1__win"] == 1.0
    # prob surprise = win - p = 1 - 0.25 = 0.75
    assert np.isclose(cur["h1__prob_surprise"], 0.75)
    # econ surprise = Y - muW = 3.0 - 1.0 = 2.0
    assert np.isclose(cur["h1__econ_surprise"], 2.0)
    # delta_p = current p - historical p = 0.50 - 0.25 = 0.25
    assert np.isclose(cur["h1__delta_p"], 0.25)


# --------------------------------------------------------------------------- #
# I. Current-outcome leakage                                                   #
# --------------------------------------------------------------------------- #
def test_current_outcome_leakage_none():
    rows = [
        mk("AG", "LONG", 10, "2024-01-01 09:00", "2024-01-02 09:00",
           1.0, 0.40, 1.0, 0.5),
        mk("AG", "LONG", 20, "2024-01-03 09:00", "2024-01-04 09:00",
           1.0, 0.50, 1.2, 0.5),
    ]
    df = make_ledger(rows)
    before = build_candidate_event_history(df).iloc[-1]
    mut = df.copy()
    # mutate CURRENT episode_return_atr AFTER features built
    mut.loc[mut.index[-1], "episode_return_atr"] = -7.0
    after = build_candidate_event_history(mut).iloc[-1]
    feat_cols = [c for c in before.index if c.startswith("h") or c in STATIC]
    for c in feat_cols:
        b, a = before[c], after[c]
        if pd.isna(b) and pd.isna(a):
            continue
        assert b == a or (isinstance(b, float) and np.isclose(b, a)), c


# --------------------------------------------------------------------------- #
# J. Feature schema + stable hashes                                            #
# --------------------------------------------------------------------------- #
def _col_hash(cols):
    return hashlib.md5(",".join(cols).encode()).hexdigest()


def test_feature_schema_and_hashes():
    assert ARMS["S0"] == STATIC
    assert ARMS["S3P"] == STATIC + M.pred_history_cols(3)
    assert ARMS["S3F"] == STATIC + M.pred_history_cols(3) + M.outcome_history_cols(3)
    assert ARMS["S5F"] == STATIC + M.pred_history_cols(5) + M.outcome_history_cols(5)
    # stable hashes (deterministic)
    h0 = _col_hash(ARMS["S0"]); h3p = _col_hash(ARMS["S3P"])
    h3f = _col_hash(ARMS["S3F"]); h5f = _col_hash(ARMS["S5F"])
    assert h0 == _col_hash(ARMS["S0"])
    assert h3p == _col_hash(ARMS["S3P"])
    assert h3f == _col_hash(ARMS["S3F"])
    assert h5f == _col_hash(ARMS["S5F"])
    # no overlap between pred-history and outcome-history feature names
    assert set(M.pred_history_cols(3)).isdisjoint(set(M.outcome_history_cols(3)))
    assert len(ARMS["S5F"]) == len(set(ARMS["S5F"]))


# --------------------------------------------------------------------------- #
# K. Meta-fold causality                                                       #
# --------------------------------------------------------------------------- #
def test_meta_fold_causality():
    rows = [
        mk("AG", "LONG", 10, "2024-01-01", "2024-01-02", 1.0, 0.5, 1.0, 0.5, fold=0),
        mk("AG", "LONG", 20, "2024-01-03", "2024-01-04", -1.0, 0.5, 1.0, 0.5, fold=0),
        mk("AG", "LONG", 30, "2024-01-10", "2024-01-11", 2.0, 0.5, 1.0, 0.5, fold=1),
        mk("AG", "LONG", 40, "2024-01-12", "2024-01-13", -2.0, 0.5, 1.0, 0.5, fold=1),
        mk("AG", "LONG", 50, "2024-02-01", "2024-02-02", 0.5, 0.5, 1.0, 0.5, fold=2),
    ]
    df = make_ledger(rows)
    splits = meta_fold_splits(df)
    eval_folds = [k for k, _, _ in splits]
    assert eval_folds == [1, 2]
    for k, tr, ev in splits:
        tr_set = set(np.where(tr)[0]); ev_set = set(np.where(ev)[0])
        assert tr_set.isdisjoint(ev_set)
        assert (df.iloc[np.where(tr)[0]]["fold"] < k).all()
        T = pd.Timestamp(sorted(df[df["fold"] == k]["decision_time"])[0])
        tr_rows = df.iloc[np.where(tr)[0]]
        assert (pd.to_datetime(tr_rows["label_available_time"]) < T).all()


# --------------------------------------------------------------------------- #
# L. Base-model fit prohibition + fit count (small real run)                    #
# --------------------------------------------------------------------------- #
def test_base_model_fit_prohibited_and_fit_count():
    reset_counters()
    res = run_t1(symbols=("AG", "AU"))
    c = res["counters"]
    assert c["base_model_fit_count"] == 0
    assert c["correction_model_fit_count"] == 32
    assert c["hyperparameter_search_count"] == 0
    assert c["candidate_history_build_count"] == 1
    assert c["test_label_read_count"] == 0
    assert set(res["meta_rows"].keys()) == {1, 2, 3, 4}
    assert res["pred_store_eval_nonnull_win_S3F"] > 0


# --------------------------------------------------------------------------- #
# M. TEST-label prohibition                                                    #
# --------------------------------------------------------------------------- #
def test_no_test_label_read():
    reset_counters()
    run_t1(symbols=("AG", "AU"))
    assert M.counters()["test_label_read_count"] == 0


# --------------------------------------------------------------------------- #
# Extra: residual targets + ledger invariants                                  #
# --------------------------------------------------------------------------- #
def test_residual_targets_and_ledger_invariants():
    # small real ledger for AG only (fast enough)
    reset_counters()
    led = build_candidate_ledger(symbols=["AG"])
    assert led.duplicated(subset=M.KEY).sum() == 0
    # history build bumps exactly once
    assert M.counters()["candidate_history_build_count"] == 0
    led = build_candidate_event_history(led)
    assert M.counters()["candidate_history_build_count"] == 1
    r_win, r_loss = residual_targets(led)
    y = led["episode_return_atr"].to_numpy(float)
    mw = led["mu_win"].to_numpy(float)
    ml = led["mu_loss"].to_numpy(float)
    win = y > 0
    assert np.allclose(r_win[win], y[win] - mw[win], equal_nan=True)
    assert np.allclose(r_loss[~win], (-y[~win]) - ml[~win], equal_nan=True)
    # rows without OOF prediction have NaN mu -> residual NaN
    no_oof = led["fold"].isna().to_numpy()
    assert np.all(np.isnan(r_win[no_oof]))


@pytest.mark.parametrize("syms", [("AG",), ("AG", "AU")])
def test_run_t1_symbol_subset(syms):
    reset_counters()
    res = run_t1(symbols=syms)
    assert res["counters"]["correction_model_fit_count"] == 32
    assert res["meta_rows"].keys() == {1, 2, 3, 4}
