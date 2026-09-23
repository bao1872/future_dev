"""Tests for direction_gated_experts_v1 (FUTURE-R4-M15-DIRECTION-GATED-EXPERTS-V1).

Covers the 33-point TEST CONTRACT. The full chain (pooled + 15-fold LOSO) is expensive,
so it runs ONCE as a session-scoped fixture; the router OOF audit is built once as a
module-scoped fixture. Everything else is a cheap deterministic correctness check.
"""

import inspect
import os

import numpy as np
import pandas as pd
import pytest

import research.liquidity_oracle_atlas.direction_gated_experts_v1 as M
from research.liquidity_oracle_atlas.train_direction_model_15sym_v1 import (
    SYMBOLS,
    verify_manifest,
)

# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def run():
    return M.run_gated_experts(save=True, verbose=False)


@pytest.fixture(scope="module")
def ctx():
    split = M.build_frozen_split()
    ds = split["ds"]
    return {"split": split, "ds": ds, "data": M.build_direction_expert_data(ds)}


@pytest.fixture(scope="module")
def oof(ctx):
    """Build the prequential OOF gates once (several fittings) and reuse for audits."""
    data = ctx["data"]
    split = ctx["split"]
    return M.build_prequential_router_oof(
        data.X9, data.y, data.w, data.decision_time_ns, split["train_idx"],
        50, return_audit=True)


# --------------------------------------------------------------------------- #
# 1. provenance / 2. manifest                                                  #
# --------------------------------------------------------------------------- #
def test_task_provenance():
    assert M.TASK_ID == "FUTURE-R4-M15-DIRECTION-GATED-EXPERTS-V1"
    assert M.BASE_SHA == "89f655be47794f06146daef0573f8ddcbfdaa7b2"
    assert M.TEST_STATUS == "architecture_diagnostic_not_pristine_confirmation"


def test_manifest_fail_closed():
    report = verify_manifest(SYMBOLS)
    assert isinstance(report, dict)
    for s in SYMBOLS:
        assert report[s].get("dataset_sha256_match") is True


# --------------------------------------------------------------------------- #
# 3. frozen TEST universe                                                      #
# --------------------------------------------------------------------------- #
def test_frozen_test_universe(ctx):
    split = ctx["split"]
    data = ctx["data"]
    te = split["test_idx"]
    gid = data.gid[te]
    n_long = int(np.unique(gid[data.y[te] == 1]).size)
    n_short = int(np.unique(gid[data.y[te] == 0]).size)
    assert te.size == 13773
    assert np.unique(gid).size == 638
    assert n_long == 319
    assert n_short == 319


# --------------------------------------------------------------------------- #
# 4. A exact reproduction (every frozen number)                                #
# --------------------------------------------------------------------------- #
def test_a_exact_reproduction(run):
    a = run["summary"]["pooled"]["A"]
    for k in ("return_atr", "accuracy", "roc_auc", "long_recall", "short_recall",
              "pred_long_share", "long_return", "short_return"):
        got = a.get(k)
        assert got is not None, f"A is missing required metric {k}"
        assert abs(got - M.A_REFERENCE[k]) < 1e-6, f"A drift on {k}: {got}"
    assert a["roc_auc_scope"] == "direction"


# --------------------------------------------------------------------------- #
# 5. fixed-router TEST prediction reproduces frozen A                          #
# --------------------------------------------------------------------------- #
def test_fixed_router_reproduces_a(run):
    rep = run["summary"]["A_reproduction"]
    assert rep["match"] is True
    # Hard predictions must be IDENTICAL; probabilities reproduce within LightGBM's
    # multi-threaded float noise budget.
    assert rep["router_hard_prediction_match"] is True
    assert rep["max_abs_dev_prob_fixed_router"] <= 1e-4
    assert abs(rep["return_atr"] - M.A_REFERENCE["return_atr"]) < 1e-6


# --------------------------------------------------------------------------- #
# 6. OOF temporal invariant / 7. first quarter warm-up                         #
# --------------------------------------------------------------------------- #
def test_oof_temporal_invariant(oof):
    audit = oof["audit"]
    assert len(audit) > 0
    for _, r in audit.iterrows():
        assert r["n_fit_rows"] > 0
        assert r["fit_has_both_classes"] is True
        assert pd.Timestamp(r["fit_max_time"]) < pd.Timestamp(r["pred_min_time"])


def test_oof_first_quarter_is_warmup(oof, ctx):
    data = ctx["data"]
    split = ctx["split"]
    tr = split["train_idx"]
    t = pd.to_datetime(data.decision_time_ns)
    pidx = pd.PeriodIndex(t, freq="Q")
    qord = (pidx.year * 4 + (pidx.quarter - 1)).to_numpy()
    first_q = np.unique(qord[tr])[0]
    first_rows = tr[qord[tr] == first_q]
    # warm-up quarter receives NO fake OOF prediction
    assert np.all(oof["pred"][first_rows] == -1)
    assert np.all(np.isnan(oof["p_long"][first_rows]))
    # every expert-trainable row has a valid OOF gate
    avail = oof["available"][tr]
    assert np.all(oof["pred"][tr][avail] >= 0)
    assert int(avail.sum()) < tr.size, "warm-up quarter must be excluded"


def test_oof_expert_rows_are_oof_gated(oof, ctx):
    """Expert TRAIN uses only rows with a valid OOF router prediction (test 8)."""
    split = ctx["split"]
    tr = split["train_idx"]
    avail = oof["available"][tr]
    usable = tr[avail]
    # all usable rows lie strictly after the warm-up quarter
    t = ctx["data"].decision_time_ns
    assert np.max(t[tr[~avail]]) < np.min(t[usable]), (
        "warm-up block must precede every OOF-gated row")


# --------------------------------------------------------------------------- #
# 9/10. Teacher absent from meta features; ROUTER side controls orientation    #
# --------------------------------------------------------------------------- #
def test_no_teacher_or_future_fields_in_meta_features():
    forbidden = ("teacher", "oracle", "bars", "entry_quality", "label",
                 "candidate_episode", "candidate_trigger", "sample_weight",
                 "dp_", "q_f1")
    for c in M.META10_COLS + M.META34_COLS:
        lc = c.lower()
        for b in forbidden:
            assert b not in lc, f"forbidden token '{b}' in feature '{c}'"


def test_router_side_not_teacher_controls_orientation():
    rng = np.random.default_rng(0)
    X9 = (rng.random((4, 9)).astype(np.float32) - 0.5)
    X33 = (rng.random((4, 33)).astype(np.float32) - 0.5)
    m10_a, m34_a, rl_a = M.build_meta_features(X9, X33, np.ones(4, np.int8),
                                               np.full(4, 0.8))
    m10_b, m34_b, rl_b = M.build_meta_features(X9, X33, np.zeros(4, np.int8),
                                               np.full(4, 0.8))
    assert rl_a.all() and not rl_b.any()
    # same input, different ROUTER side -> different orientation (proves router drives it)
    assert not np.array_equal(m10_a, m10_b)
    assert not np.array_equal(m34_a, m34_b)
    # orientation must equal negation for the signed DTP9 block
    assert np.allclose(m10_a[:, :9], -m10_b[:, :9])


# --------------------------------------------------------------------------- #
# 11. DTP9 side transform synthetic                                            #
# --------------------------------------------------------------------------- #
def test_dtp9_side_transform_synthetic():
    X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]], dtype=np.float32)
    long = np.array([True])
    short = np.array([False])
    assert np.allclose(M.orient_dtp9_router_side(X, long), X)
    assert np.allclose(M.orient_dtp9_router_side(X, short), -X)


# --------------------------------------------------------------------------- #
# 12/13. STRUCT33 LONG orientation / SHORT swap                                #
# --------------------------------------------------------------------------- #
def _one_row(vals):
    row = np.zeros(33, dtype=np.float32)
    for k, v in vals.items():
        row[k] = v
    return row[None, :]


def _base_vals():
    return {
        0: 0.5, 1: 0.3, 2: 0.1,
        3: 1.0, 4: 2.0,
        5: 3.0, 6: 4.0,
        7: 5.0, 8: 6.0,
        9: 7.0, 10: 8.0,
    }


def test_struct33_long_orientation():
    X = _one_row(_base_vals())
    out = M.orient_struct33_router_side(X, np.array([True]))
    assert out[0, 0] == 0.5 and out[0, 1] == 0.3 and out[0, 2] == 0.1
    assert out[0, 3] == 1.0 and out[0, 4] == 2.0
    assert out[0, 5] == 3.0 and out[0, 6] == 4.0
    assert out[0, 7] == 5.0 and out[0, 8] == 6.0
    assert out[0, 9] == 7.0 and out[0, 10] == 8.0


def test_struct33_short_swap():
    X = _one_row(_base_vals())
    out = M.orient_struct33_router_side(X, np.array([False]))
    assert out[0, 0] == -0.5 and out[0, 1] == -0.3 and out[0, 2] == -0.1
    assert out[0, 3] == 2.0 and out[0, 4] == 1.0      # support <-> resistance
    assert out[0, 5] == 4.0 and out[0, 6] == 3.0      # strengths swapped
    assert out[0, 7] == 6.0 and out[0, 8] == 5.0      # liq_up <-> liq_down
    assert out[0, 9] == 8.0 and out[0, 10] == 7.0     # counts swapped


# --------------------------------------------------------------------------- #
# 14. router_p_side formula                                                    #
# --------------------------------------------------------------------------- #
def test_router_p_side_formula():
    p_long = np.array([0.9, 0.7, 0.5, 0.2, 0.1])
    router_long = np.array([True, True, True, False, False])
    got = M.router_confidence(p_long, router_long)
    expected = np.array([0.9, 0.7, 0.5, 0.8, 0.9])
    assert np.allclose(got, expected)
    assert got.min() >= 0.5 and got.max() <= 1.0


# --------------------------------------------------------------------------- #
# 15/16. META10 exactly 10, META34 exactly 34                                  #
# --------------------------------------------------------------------------- #
def test_meta_feature_counts():
    assert len(M.META10_COLS) == 10
    assert len(M.META34_COLS) == 34
    rng = np.random.default_rng(1)
    X9 = rng.random((6, 9)).astype(np.float32)
    X33 = rng.random((6, 33)).astype(np.float32)
    rp = np.array([1, 1, 0, 0, 1, 0], dtype=np.int8)
    pl = np.array([0.8, 0.6, 0.55, 0.9, 0.7, 0.51])
    m10, m34, rl = M.build_meta_features(X9, X33, rp, pl)
    assert m10.shape == (6, 10)
    assert m34.shape == (6, 34)
    assert np.array_equal(rl, rp == 1)


# --------------------------------------------------------------------------- #
# 17/18. one shared model vs two distinct expert objects                       #
# --------------------------------------------------------------------------- #
def test_shared_corrector_fits_exactly_one_model(monkeypatch):
    """M9 / M33 must be ONE shared classifier (test 17)."""
    captured = []

    def spy(Xtr, ytr, wtr, Xv, yv, wv):
        captured.append(np.asarray(Xtr))

        class _CM:
            def predict_proba(self, X):
                return np.tile(np.array([0.4, 0.6]), (len(X), 1))
        return _CM()

    monkeypatch.setattr(M, "fit_direction_model", spy)
    rng = np.random.default_rng(4)
    Xtr = rng.random((8, 10)).astype(np.float32)
    ytr = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
    gid_tr = np.array([f"t{i//2}" for i in range(8)])
    Xva = rng.random((4, 10)).astype(np.float32)
    yva = np.array([1, 0, 1, 0], dtype=np.uint8)
    gid_va = np.array(["v0", "v0", "v1", "v1"])
    M.fit_shared_corrector(Xtr, ytr, gid_tr, Xva, yva, gid_va)
    assert len(captured) == 1, "shared corrector must fit exactly ONE model"


# --------------------------------------------------------------------------- #
# 19/20/21. expert populations are router-gated and two-class                  #
# --------------------------------------------------------------------------- #
def test_expert_populations_are_router_gated(monkeypatch):
    captured = []

    def spy(Xtr, ytr, wtr, Xv, yv, wv):
        captured.append({"Xtr": np.asarray(Xtr).copy(), "ytr": np.asarray(ytr).copy()})

        class _CM:
            def predict_proba(self, X):
                return np.tile(np.array([0.4, 0.6]), (len(X), 1))
        return _CM()

    monkeypatch.setattr(M, "fit_direction_model", spy)

    rng = np.random.default_rng(2)
    X = rng.random((10, 10)).astype(np.float32)
    y_meta = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0], dtype=np.uint8)
    gid = np.array([f"t{i//2}" for i in range(10)])
    router_tr = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1], dtype=np.int8)
    Xv = rng.random((6, 10)).astype(np.float32)
    y_meta_v = np.array([1, 0, 1, 0, 1, 0], dtype=np.uint8)
    gid_v = np.array([f"v{i//2}" for i in range(6)])
    router_va = np.array([1, 1, 0, 0, 1, 0], dtype=np.int8)

    long_model, short_model = M.fit_expert_pair(
        X, y_meta, gid, router_tr, Xv, y_meta_v, gid_v, router_va)
    assert len(captured) == 2, "exactly two experts fitted"
    assert long_model is not None and short_model is not None
    assert long_model is not short_model, "E9/E33 must be TWO distinct fitted models"
    long_X, short_X = captured[0]["Xtr"], captured[1]["Xtr"]
    assert np.array_equal(long_X, X[router_tr == 1]), "LongExpert saw non-LONG rows"
    assert np.array_equal(short_X, X[router_tr == 0]), "ShortExpert saw non-SHORT rows"
    assert np.array_equal(captured[0]["ytr"], y_meta[router_tr == 1])
    assert np.array_equal(captured[1]["ytr"], y_meta[router_tr == 0])


def test_expert_single_class_population_fails_closed(monkeypatch):
    def spy(Xtr, ytr, wtr, Xv, yv, wv):
        class _CM:
            def predict_proba(self, X):
                return np.tile(np.array([0.4, 0.6]), (len(X), 1))
        return _CM()

    monkeypatch.setattr(M, "fit_direction_model", spy)
    rng = np.random.default_rng(3)
    X = rng.random((4, 10)).astype(np.float32)
    gid = np.array(["a", "a", "b", "b"])
    router = np.array([1, 1, 0, 0], dtype=np.int8)
    y_meta = np.ones(4, dtype=np.uint8)  # single class -> must fail closed
    with pytest.raises(AssertionError):
        M.fit_expert_pair(X, y_meta, gid, router, X, y_meta, gid, router)


# --------------------------------------------------------------------------- #
# 22. equal-trade training weights sum to 1                                    #
# --------------------------------------------------------------------------- #
def test_equal_trade_weights_sum_to_one():
    gid = np.array(["t1", "t1", "t1", "t2", "t3", "t3"])
    w = M.equal_trade_weights(gid)
    assert np.isclose(w[gid == "t1"].sum(), 1.0)
    assert np.isclose(w[gid == "t2"].sum(), 1.0)
    assert np.isclose(w[gid == "t3"].sum(), 1.0)
    assert np.allclose(w[:3], 1 / 3)
    assert np.allclose(w[4:], 0.5)


# --------------------------------------------------------------------------- #
# 23. no TEST row enters any fit                                               #
# --------------------------------------------------------------------------- #
def test_no_test_row_in_train_or_val(ctx):
    split = ctx["split"]
    tr, va, te = split["train_idx"], split["val_idx"], split["test_idx"]
    assert len(set(tr.tolist()) & set(te.tolist())) == 0
    assert len(set(va.tolist()) & set(te.tolist())) == 0
    assert len(set(tr.tolist()) & set(va.tolist())) == 0


# --------------------------------------------------------------------------- #
# 24. keep/flip rule uses the fixed 0.5 threshold                              #
# --------------------------------------------------------------------------- #
def test_keep_flip_rule():
    router = np.array([1, 1, 0, 0], dtype=np.int8)
    p_correct = np.array([0.5, 0.49, 0.5, 0.51])
    final = M.apply_shared_corrector(router, p_correct)
    # p_correct >= 0.5 keeps (so 0.5 and 0.51 both keep); < 0.5 flips
    assert np.array_equal(final, np.array([1, 0, 0, 0], dtype=np.uint8))
    # only a strictly-below-0.5 score flips, for either router side
    router2 = np.array([1, 0], dtype=np.int8)
    assert np.array_equal(M.apply_shared_corrector(router2, np.array([0.50, 0.49])),
                          np.array([1, 1], dtype=np.uint8))


# --------------------------------------------------------------------------- #
# 25. no threshold search / tuning tokens                                      #
# --------------------------------------------------------------------------- #
def test_no_threshold_or_hyperparameter_search():
    src = inspect.getsource(M)
    for tok in ("GridSearchCV", "RandomizedSearchCV", "best_threshold",
                "threshold_search", "optuna", "learning_rate_="):
        assert tok not in src, f"forbidden tuning token present: {tok}"
    # the decision rule must be the literal 0.5 constant
    dec = inspect.getsource(M.apply_shared_corrector)
    assert "0.5" in dec


# --------------------------------------------------------------------------- #
# 26. EntryQuality / bars / oracle future fields absent from X                 #
# --------------------------------------------------------------------------- #
def test_no_entryquality_or_bars_in_meta_arrays(ctx):
    data = ctx["data"]
    te = ctx["split"]["test_idx"]
    rp = np.ones(te.size, dtype=np.int8)
    pl = np.full(te.size, 0.7)
    m10, m34, _ = M.build_meta_features(data.X9[te], data.X33[te], rp, pl)
    # the last column is router_p_side in [0.5,1]; no column may equal entry_quality_atr
    eq = data.entry_quality_atr[te]
    for k in range(m10.shape[1]):
        assert not np.array_equal(m10[:, k], eq)
    for k in range(m34.shape[1]):
        assert not np.array_equal(m34[:, k], eq)


# --------------------------------------------------------------------------- #
# 27/28/29. sample_weight_raw evaluation + aligned trade ids + delta arithmetic #
# --------------------------------------------------------------------------- #
def test_trade_returns_aligned_and_delta_arithmetic(run):
    df = pd.read_csv(run["paths"]["trade_returns_csv"])
    assert len(df) == 638
    assert df["gid"].is_unique
    for pair in (("A", "E9"), ("M9", "E9"), ("E9", "E33"), ("M33", "E33"),
                 ("A", "M9"), ("M9", "M33"), ("A", "E33")):
        ka, kb = pair
        assert np.allclose(df[f"delta_{kb}_minus_{ka}"],
                           df[f"tr_{kb}"] - df[f"tr_{ka}"], atol=1e-9)


def test_delta_arithmetic_exact(run):
    df = pd.read_csv(run["paths"]["trade_returns_csv"])
    assert np.allclose(df["delta_E9_minus_A"], df["tr_E9"] - df["tr_A"], atol=1e-12)


# --------------------------------------------------------------------------- #
# 30. repair / damage synthetic                                                #
# --------------------------------------------------------------------------- #
def test_repair_damage_synthetic():
    router = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.uint8)
    y = np.array([1, 1, 0, 0, 0, 0, 1, 1], dtype=np.uint8)
    # rows 0,1 correct already; rows 2,3 wrong; rows 4,5 correct; 6,7 wrong
    final = np.array([1, 1, 0, 1, 0, 0, 0, 1], dtype=np.uint8)
    # row3 repaired (wrong->correct); row0,1 keep correct; row 7 repaired
    # no damage introduced
    w = np.ones(8)
    got = M._repair_damage(router, final, y, w)
    assert got["repair_rate"] == pytest.approx(2 / 4)   # 2 of 4 wrong rows repaired
    assert got["damage_rate"] == pytest.approx(0.0)
    assert got["flip_rate"] == pytest.approx(2 / 8)


def test_repair_damage_weighted():
    router = np.array([1, 1, 0, 0], dtype=np.uint8)
    y = np.array([1, 0, 0, 1], dtype=np.uint8)
    final = np.array([1, 0, 0, 0], dtype=np.uint8)
    w = np.array([1.0, 3.0, 1.0, 5.0])
    got = M._repair_damage(router, final, y, w)
    # A wrong on idx1 (w=3) and idx3 (w=5); only idx1 becomes correct -> repaired
    assert got["repair_rate"] == pytest.approx(3.0 / 8.0)
    # A right on idx0 (w=1) and idx2 (w=1); neither becomes wrong -> no damage
    assert got["damage_rate"] == pytest.approx(0.0)
    # only idx1 flipped, and it carried weight 3 of 10
    assert got["flip_rate"] == pytest.approx(3.0 / 10.0)


# --------------------------------------------------------------------------- #
# 31. LOSO held-out symbol absent from TRAIN + VAL                             #
# --------------------------------------------------------------------------- #
def test_loso_held_out_absent(ctx, run):
    sym = ctx["ds"]["symbol"].to_numpy(object)
    split = ctx["split"]
    lo = pd.read_csv(run["paths"]["loso_csv"])
    assert len(lo) == 15
    assert set(lo["held_out_symbol"]) == set(SYMBOLS)
    for s in SYMBOLS:
        tr = split["train_idx"][sym[split["train_idx"]] != s]
        va = split["val_idx"][sym[split["val_idx"]] != s]
        assert (sym[tr] == s).sum() == 0
        assert (sym[va] == s).sum() == 0


# --------------------------------------------------------------------------- #
# 32. time blocks disjoint and cover all opportunities                         #
# --------------------------------------------------------------------------- #
def test_time_blocks_cover_all_opportunities(run):
    tb = pd.read_csv(run["paths"]["time_blocks_csv"])
    assert len(tb) > 0
    assert tb["month"].is_unique
    assert int(tb["n_opportunities"].sum()) == 638


# --------------------------------------------------------------------------- #
# 33. evidence schema complete                                                 #
# --------------------------------------------------------------------------- #
def test_evidence_schema_complete(run):
    s = run["summary"]
    for key in ("task_id", "base_sha", "test_status", "model_roles", "frozen_split",
                "A_reproduction", "router", "features", "contract", "pooled",
                "pooled_ex_ag", "paired_contrasts", "side_contrasts", "per_symbol",
                "per_symbol_cluster_bootstrap", "loso", "time_blocks", "verdict",
                "provenance"):
        assert key in s, f"missing {key}"
    for key in ("E9_minus_A", "E9_minus_M9", "E33_minus_E9", "E33_minus_M33"):
        assert key in s["paired_contrasts"], f"missing contrast {key}"
        assert key in s["side_contrasts"], f"missing side contrast {key}"
        grp = {"E9_minus_A": "expert_total_increment",
               "E9_minus_M9": "expert_split_increment",
               "E33_minus_E9": "struct33_within_experts_increment",
               "E33_minus_M33": "struct33_specialist_increment"}[key]
        assert s["verdict"][grp]["state"] in (
            "supported_diagnostically", "harmful_diagnostically",
            "no_identifiable_increment")
    for p in run["paths"].values():
        assert os.path.exists(p), p


def test_all_systems_reported_and_phase_present(run):
    pooled = run["summary"]["pooled"]
    for s in M.SYSTEMS:
        assert s in pooled
        assert "phase" in pooled[s]
        for ph in ("BEFORE_ENTRY", "AT_ENTRY", "IN_POSITION"):
            assert ph in pooled[s]["phase"]
    # LOSO and per-symbol present
    assert len(run["summary"]["loso"]["folds"]) == 15
    assert len(run["summary"]["per_symbol"]) == 15
