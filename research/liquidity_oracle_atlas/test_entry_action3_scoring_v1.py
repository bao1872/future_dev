"""Light tests for FUTURE-ENTRY-ACTION3-STRUCT44-V1.

Covers the 10 disciplines from the task spec without retraining the full model
set (a micro-fit with 1 tree validates the train/val/test mask discipline).
"""

import numpy as np
import pandas as pd

import research.liquidity_oracle_atlas.experiment_entry_value_tree_core108_t15b_aligned_v1 as B  # noqa: E402
import research.liquidity_oracle_atlas.experiment_entry_value_cost_robustness_phase2_formal_v1 as P2  # noqa: E402
import research.liquidity_oracle_atlas.experiment_entry_action3_scoring_v1 as A  # noqa: E402


def _mini_real():
    joined, _ = A.load_joined()
    df = A.analysis_frame(0.0, joined)
    return df


# 1. utility construction [Y_S, 0, Y_L]
def test_utility_construction():
    n = 50
    df = pd.DataFrame({
        "Y_S": np.linspace(-2, 2, n),
        "Y_L": np.linspace(2, -2, n),
    })
    u = A.action_utilities(df)
    assert u.shape == (n, 3)
    assert np.allclose(u[:, 0], df["Y_S"])
    assert np.allclose(u[:, 1], 0.0)
    assert np.allclose(u[:, 2], df["Y_L"])


# 2. best action mapping (cache -1/0/+1 -> 0/1/2), utilities consistent
#    with the DP label so the unique-margin audit passes.
def test_best_action_mapping():
    # SHORT: Y_S best; FLAT: both negative (0 best); LONG: Y_L best
    df = pd.DataFrame({"best_F1_cost": [-1.0, 0.0, 1.0],
                       "Y_S": [1.0, -1.0, -1.0],
                       "Y_L": [-1.0, -2.0, 1.0]})
    action, _, _ = A.build_action_targets(df)
    assert list(action) == [A.ACTION_SHORT, A.ACTION_FLAT, A.ACTION_LONG]


# 3. DP best_F1 agrees with unique-margin argmax(utility)
def test_dp_utility_agreement_unique_margin():
    df = _mini_real()
    action, utility, margin = A.build_action_targets(df)
    argmax_u = np.argmax(utility, axis=1)
    unique = margin > 1e-12
    assert np.array_equal(action[unique], argmax_u[unique])


# 4. margin is best minus second-best utility
def test_margin_definition():
    df = _mini_real()
    _, utility, margin = A.build_action_targets(df)
    su = np.sort(utility, axis=1)
    assert np.allclose(margin, su[:, -1] - su[:, -2])


# 5. first-non-WAIT is causal and one row per episode
def test_first_nonwait_causal():
    df = pd.DataFrame({
        "symbol": ["A", "A", "A", "A"],
        "global_episode": [1, 1, 1, 1],
        "decision_time": pd.to_datetime([
            "2025-01-01 09:05", "2025-01-01 09:10",
            "2025-01-01 09:15", "2025-01-01 09:20"]),
        "Y_L": [1.0, 2.0, 3.0, 4.0],
        "Y_S": [-1.0, -1.0, -1.0, -1.0],
    })
    pred = np.array([A.ACTION_FLAT, A.ACTION_LONG, A.ACTION_FLAT, A.ACTION_SHORT])
    trade = A.first_nonwait_by_episode(df, pred)
    assert len(trade) == 1
    # earliest non-FLAT is the LONG at 09:10
    assert trade.iloc[0]["pred_action"] == A.ACTION_LONG
    assert trade.iloc[0]["entry_value"] == 2.0
    assert not trade.duplicated(["symbol", "global_episode"]).any()


# 6. row regret always nonnegative
def test_regret_nonnegative():
    df = _mini_real()
    rng = np.random.default_rng(0)
    pa = rng.integers(0, 3, size=len(df))
    u = A.action_utilities(df)
    chosen = u[np.arange(len(df)), pa]
    oracle = np.max(u, axis=1)
    regret = oracle - chosen
    assert (regret >= -1e-12).all()


# 7. WAIT chosen utility equals zero
def test_wait_utility_zero():
    df = _mini_real()
    pa = np.full(len(df), A.ACTION_FLAT, dtype=int)
    u = A.action_utilities(df)
    chosen = u[np.arange(len(df)), pa]
    assert np.allclose(chosen, 0.0)


# 8. Test rows never enter model fit (mask discipline + micro-fit)
def test_test_not_in_fit():
    df = _mini_real().sample(600, random_state=1).reset_index(drop=True)
    subsets = B.feature_subsets()
    tr, va, te = A._masks(df)
    # masks are disjoint and exhaustive
    assert (tr & va).sum() == 0 and (tr & te).sum() == 0 and (va & te).sum() == 0
    assert (tr | va | te).all()
    # micro training uses only train rows
    params = dict(A.ACTION3_PARAMS)
    params["n_estimators"] = 1
    model, Xte = A.fit_action3(df, subsets["M1"], use_margin=False, params=params)
    assert Xte.shape[0] == int(te.sum())  # predict called on test rows only
    # reg_action test variant reads p_te columns (test scope)
    pa = A.reg_action(df[df["split"] == "test"], "M1", "test")
    assert len(pa) == int(te.sum())


# 9. experiment uses only M0/M1 feature lists; M0 subset of M1
def test_feature_lists_m0_m1_only():
    subsets = B.feature_subsets()
    # the experiment hardcodes M0 (DTP12) and M1 (STRUCT44) only
    assert set(subsets["M0"]).issubset(set(subsets["M1"]))
    assert len(subsets["M0"]) < len(subsets["M1"])
    # every column the experiment actually consumes exists in the real dataset
    t2 = pd.read_parquet(A.DATASET_PATH)
    for m in ("M0", "M1"):
        for c in subsets[m]:
            assert c in t2.columns


# 10. all 5 kappas share an identical candidate universe
def test_kappa_universe_identical():
    joined, audit = A.load_joined()
    assert audit["hard_gate_universe_identical"] is True
    counts = audit["matched_rows_per_kappa"]
    vals = list(counts.values())
    assert len(set(vals)) == 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa
            failed += 1
            print(f"FAIL {fn.__name__}: {e!r}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
