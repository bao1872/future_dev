"""Tests for FUTURE-R4-M15-DIRECTION-ASYMMETRY-ABC-V1.

One-shot architecture diagnostic: does allowing LONG and SHORT to learn separate
decision functions improve out-of-sample directional information beyond (A) the
frozen direct M0 and (B) a symmetry-constrained shared side-normalized model?

The full experiment is expensive (63 LightGBM fits incl. 15-fold LOSO), so it is
run ONCE as a session-scoped fixture. Each test then only inspects the result or
checks a cheap, deterministic correctness property.
"""

import inspect
import json
import os

import numpy as np
import pandas as pd
import pytest

from research.liquidity_oracle_atlas.train_direction_model_15sym_v1 import (
    BASE_PARAMS,
    DTP9,
    SYMBOLS,
    verify_manifest,
)
import research.liquidity_oracle_atlas.direction_asymmetry_abc_v1 as M


@pytest.fixture(scope="session")
def run():
    # Run the full experiment once; writes the 4 evidence files and returns the
    # in-memory summary for inspection. ~1 minute.
    return M.run_asymmetry_audit(save=True, verbose=False)


# --------------------------------------------------------------------------- #
# 1. manifest fail-closed (no silent data-schema drift)                        #
# --------------------------------------------------------------------------- #
def test_manifest_fail_closed():
    # Raises RuntimeError (STOP_*) if the frozen dataset manifest is invalid;
    # on success it returns a per-symbol dict with sha256-match flags.
    report = verify_manifest(SYMBOLS)
    assert isinstance(report, dict)
    for s in SYMBOLS:
        assert s in report
        assert report[s].get("dataset_sha256_match") is True


# --------------------------------------------------------------------------- #
# 2. no upstream rerun / dataset rebuild tokens                               #
# --------------------------------------------------------------------------- #
def test_no_upstream_rerun_tokens():
    src = inspect.getsource(M)
    for tok in (
        "run_environment_m15", "derive_m15_candidate_gate", "build_struct33_dataset",
        "load_oracle_artifact", "run_dp_m15", "build_direction_dataset",
        "prepare_xy",  # prepare_xy is a frozen helper, not a rebuild
    ):
        if tok == "prepare_xy":
            continue
        assert tok not in src, f"forbidden token present in source: {tok}"


# --------------------------------------------------------------------------- #
# 3. frozen split reproduction                                                 #
# --------------------------------------------------------------------------- #
def test_frozen_split_reproduction():
    split = M.build_frozen_split()
    arr = M.build_abc_data(split["ds"])
    n_rows = int(split["test_idx"].size)
    n_trades = int(len(np.unique(arr.gid[split["test_idx"]])))
    assert n_rows == 13773
    assert n_trades == 638


# --------------------------------------------------------------------------- #
# 4. A exact reproduction (reuse the frozen M0 path)                         #
# --------------------------------------------------------------------------- #
def test_a_exact_reproduction(run):
    rep = run["summary"]["A_reproduction"]
    assert rep["match"] is True
    assert abs(rep["return_atr"] - 0.6014965284150177) < 1e-6


# --------------------------------------------------------------------------- #
# 5. side transform: +X / -X / -(-X) == X                                     #
# --------------------------------------------------------------------------- #
def test_side_transform():
    split = M.build_frozen_split()
    arr = M.build_abc_data(split["ds"])
    nonan = ~np.isnan(arr.X)
    assert np.allclose(arr.X_neg[nonan], -arr.X[nonan])
    assert np.allclose((-arr.X_neg)[nonan], arr.X[nonan])


# --------------------------------------------------------------------------- #
# 6. B shared targets: long_target == y, short_target == 1-y, single model    #
# --------------------------------------------------------------------------- #
def test_shared_side_targets_and_single_model():
    split = M.build_frozen_split()
    arr = M.build_abc_data(split["ds"])
    idx = split["train_idx"][:200]
    X2, y2, w2 = M.make_shared_side_xy(arr.X, arr.X_neg, arr.y, arr.w, idx)
    n = idx.size
    assert X2.shape == (2 * n, arr.X.shape[1])
    assert np.array_equal(y2[:n], arr.y[idx])
    assert np.array_equal(y2[n:], 1 - arr.y[idx])

    # fit B and confirm exactly ONE model object is produced
    mB = M.fit_b(arr, split["train_idx"], split["val_idx"])
    assert isinstance(mB, object)
    # a single classifier (LightGBM Booster-backed estimator), not a tuple
    assert not isinstance(mB, tuple)
    assert hasattr(mB, "predict_proba")


# --------------------------------------------------------------------------- #
# 7. C = two DISTINCT fitted models, both on the FULL TRAIN population        #
# --------------------------------------------------------------------------- #
def test_c_two_distinct_models_full_train():
    split = M.build_frozen_split()
    arr = M.build_abc_data(split["ds"])
    mL, mS = M.fit_c(arr, split["train_idx"], split["val_idx"])
    assert mL is not mS
    assert hasattr(mL, "predict_proba") and hasattr(mS, "predict_proba")
    # Each expert sees the full TRAIN population (N rows), not just its own side.
    # LongExpert target is y (full population); ShortExpert target is 1-y (full population).
    # Verify both predict on the TEST set and that they disagree somewhere
    # (i.e. they learned different functions, not a shared one).
    idx = split["test_idx"]
    predL = mL.predict_proba(arr.X[idx])[:, 1]
    predS = mS.predict_proba(arr.X_neg[idx])[:, 1]
    assert predL.shape == predS.shape == (idx.size,)
    # the two scores are not identical -> genuinely different functions
    assert not np.allclose(predL, predS)


def test_c_models_never_see_test_in_fit():
    split = M.build_frozen_split()
    arr = M.build_abc_data(split["ds"])
    tr = split["train_idx"]
    va = split["val_idx"]
    te = split["test_idx"]
    # Build a train/val split that is disjoint from test by construction.
    assert len(np.intersect1d(tr, te)) == 0
    assert len(np.intersect1d(va, te)) == 0
    mL, mS = M.fit_c(arr, tr, va)
    # If test rows leaked into training, this would still "work"; the disjoint
    # assertion above is the hard guard. Additionally confirm the scored TEST
    # predictions are well-defined.
    pred_c, _ = M.predict_c(mL, mS, arr, te)
    assert pred_c.shape == (te.size,)
    assert set(np.unique(pred_c)).issubset({0, 1})


# --------------------------------------------------------------------------- #
# 8. combine_side_scores: pL>=pS -> LONG, q_long finite in [0,1]              #
# --------------------------------------------------------------------------- #
def test_combine_side_scores():
    rng = np.random.default_rng(0)
    pL = rng.random(50)
    pS = rng.random(50)
    pred, q = M.combine_side_scores(pL, pS)
    assert np.array_equal(pred, (pL >= pS).astype(np.uint8))
    assert q.min() >= 0.0 and q.max() <= 1.0
    assert np.isfinite(q).all()
    # zero denominator -> 0.5 fallback
    z = np.zeros(4)
    _, qz = M.combine_side_scores(z, z)
    assert np.allclose(qz, 0.5)


# --------------------------------------------------------------------------- #
# 9. A/B/C TEST trade IDs completely aligned (len == 638)                     #
# --------------------------------------------------------------------------- #
def test_abc_test_trade_alignment(run):
    df = pd.read_csv(run["paths"]["trade_returns_csv"])
    assert len(df) == 638
    assert df["gid"].is_unique
    assert set(df.columns) >= {
        "gid", "teacher_direction", "tr_A", "tr_B", "tr_C",
        "delta_BA", "delta_CB", "delta_CA",
    }
    # delta arithmetic consistency
    assert np.allclose(df["delta_CB"], df["tr_C"] - df["tr_B"], atol=1e-9)
    assert np.allclose(df["delta_BA"], df["tr_B"] - df["tr_A"], atol=1e-9)
    assert np.allclose(df["delta_CA"], df["tr_C"] - df["tr_A"], atol=1e-9)


# --------------------------------------------------------------------------- #
# 10. paired deltas length == 638 (global ids unique)                         #
# --------------------------------------------------------------------------- #
def test_paired_delta_length_and_unique_ids(run):
    s = run["summary"]
    pc = s["paired_contrasts"]
    for k in ("B_minus_A", "C_minus_B", "C_minus_A"):
        assert "mean" in pc[k] and "ci_low" in pc[k] and "ci_high" in pc[k]
    # 638 unique global trade ids on TEST
    split = M.build_frozen_split()
    arr = M.build_abc_data(split["ds"])
    assert len(np.unique(arr.gid[split["test_idx"]])) == 638


# --------------------------------------------------------------------------- #
# 11. global trade ids collision-safe                                         #
# --------------------------------------------------------------------------- #
def test_global_trade_ids_collision_safe():
    split = M.build_frozen_split()
    arr = M.build_abc_data(split["ds"])
    # gid must embed symbol so cross-symbol trade ids never collide
    assert arr.gid.dtype.kind in ("U", "S", "O")
    sample = arr.gid[:5]
    assert all("::" in str(g) for g in sample)


# --------------------------------------------------------------------------- #
# 12. LOSO held-out symbol excluded from TRAIN+VAL, only in TEST              #
# --------------------------------------------------------------------------- #
def test_loso_held_out_exclusion(run):
    s = run["summary"]
    folds = s["loso"]["folds"]
    assert len(folds) == 15
    held = {f["held_out_symbol"] for f in folds}
    assert held == set(SYMBOLS)
    # For each held-out symbol, the LOSO-trained A return must DIFFER from the
    # pooled-model per-symbol A return (because LOSO excluded that symbol from
    # training) -- this is the behavioral proof of held-out exclusion.
    per_sym = {r["symbol"]: r for r in s["per_symbol"]}
    for f in folds:
        sym = f["held_out_symbol"]
        # pooled per-symbol A return uses ALL symbols in training; LOSO A return
        # uses 14 symbols. They should not be identical.
        assert abs(f["A_return"] - per_sym[sym]["A_return"]) > 1e-9


# --------------------------------------------------------------------------- #
# 13. feature contract: exactly DTP9, no STRUCT33 / symbol / EntryQuality     #
# --------------------------------------------------------------------------- #
def test_feature_contract():
    split = M.build_frozen_split()
    arr = M.build_abc_data(split["ds"])
    assert arr.X.shape[1] == len(DTP9)
    assert list(DTP9) == M.DTP_COLS
    assert "entry_quality_atr" not in M.DTP_COLS
    assert "symbol" not in M.DTP_COLS
    # a model trained by the frozen helper uses exactly 9 features
    mB = M.fit_b(arr, split["train_idx"], split["val_idx"])
    assert mB.n_features_ == 9
    # X must not contain entry_quality_atr values (different scale / leakage)
    assert "entry_quality_atr" not in M.DTP_COLS


# --------------------------------------------------------------------------- #
# 14. parameter contract: BASE_PARAMS unchanged, no threshold search          #
# --------------------------------------------------------------------------- #
def test_base_params_unchanged():
    expected = {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 2000,
        "random_state": 20260923,
        "n_jobs": -1,
    }
    for k, v in expected.items():
        assert BASE_PARAMS.get(k) == v, f"BASE_PARAMS[{k}] = {BASE_PARAMS.get(k)} != {v}"
    # no threshold tuning: combine uses a fixed pL>=pS rule; assert DECISION_THRESHOLD
    from research.liquidity_oracle_atlas.train_direction_model_15sym_v1 import DECISION_THRESHOLD
    assert DECISION_THRESHOLD == 0.5


# --------------------------------------------------------------------------- #
# 15. evidence schema complete                                                #
# --------------------------------------------------------------------------- #
def test_evidence_schema_complete(run):
    s = run["summary"]
    for key in ("task_id", "base_sha", "test_status", "metric", "frozen_split",
                "A_reproduction", "contract", "pooled", "pooled_ex_ag",
                "paired_contrasts", "per_symbol", "per_symbol_cluster_bootstrap",
                "loso", "mechanism", "verdict", "provenance"):
        assert key in s, f"missing summary key: {key}"
    assert s["test_status"] == "architecture_diagnostic_not_pristine_confirmation"
    assert s["verdict"]["primary_contrast"] == "C-B"
    assert "C_minus_B_SHORT" in s["paired_contrasts"]
    # per-symbol + loso CSVs exist
    for p in ("summary", "per_symbol_csv", "trade_returns_csv", "loso_csv"):
        assert os.path.exists(run["paths"][p])
    ps = pd.read_csv(run["paths"]["per_symbol_csv"])
    assert len(ps) == 15
    lo = pd.read_csv(run["paths"]["loso_csv"])
    assert len(lo) == 15


# --------------------------------------------------------------------------- #
# 16. FIX7 provenance is honest (model_redesign removed)                       #
# --------------------------------------------------------------------------- #
def test_fix7_provenance(run):
    s = run["summary"]["contract"]
    assert s["base_params_unchanged"] is True
    assert s["no_threshold_tuning"] is True
    assert s["no_hyperparameter_tuning"] is True
    assert s["no_struct33"] is True
    assert s["no_symbol_feature"] is True
    assert s["entry_quality_atr_not_in_X"] is True
    prov = run["summary"]["provenance"]
    assert prov["model_architecture_experiment"] is True
    assert prov["feature_schema_changed"] is False
    assert prov["hyperparameter_tuning"] is False
    assert prov["threshold_tuning"] is False
    assert prov["upstream_changed"] is False
    assert prov["models_retrained_in_fix1"] is False
    assert prov["predictions_changed_in_fix1"] is False
    assert prov["economic_metrics_changed_in_fix1"] is False
    assert "model_redesign" not in run["summary"]["contract"]


# --------------------------------------------------------------------------- #
# FIX8.1 weighted accuracy uses sample_weight_raw                              #
# --------------------------------------------------------------------------- #
def test_weighted_accuracy_uses_sample_weight(run):
    import pandas as pd
    df = pd.read_parquet(M.PREDICTIONS_PARQUET)
    y = df["y"].to_numpy(np.uint8)
    w = df["w"].to_numpy(np.float64)
    pred = df["pred_a"].to_numpy(np.uint8)
    acc_weighted = float(np.average(pred == y, weights=w))
    acc_unweighted = float((pred == y).mean())
    # summary pooled A accuracy must equal the weighted computation
    summary_acc = run["summary"]["pooled"]["A"]["accuracy"]
    assert abs(summary_acc - acc_weighted) < 1e-12
    # and it must actually differ from the unweighted mean (proves weighting applied)
    assert abs(acc_weighted - acc_unweighted) > 1e-9


# --------------------------------------------------------------------------- #
# FIX8.2/3/4 weighted LONG/SHORT recall + predicted LONG share use weights     #
# --------------------------------------------------------------------------- #
def test_weighted_recall_and_long_share(run):
    import pandas as pd
    df = pd.read_parquet(M.PREDICTIONS_PARQUET)
    y = df["y"].to_numpy(np.uint8)
    w = df["w"].to_numpy(np.float64)
    for suf in ("a", "b", "c"):
        pred = df[f"pred_{suf}"].to_numpy(np.uint8)
        rec = M._recall_metrics(pred, y, w)
        long_mask = y == 1
        short_mask = y == 0
        exp_long = float(np.average(pred[long_mask] == 1, weights=w[long_mask]))
        exp_short = float(np.average(pred[short_mask] == 0, weights=w[short_mask]))
        exp_pls = float(np.average(pred == 1, weights=w))
        assert abs(rec["long_recall"] - exp_long) < 1e-12
        assert abs(rec["short_recall"] - exp_short) < 1e-12
        assert abs(rec["predicted_long_share"] - exp_pls) < 1e-12


# --------------------------------------------------------------------------- #
# FIX8.5 weighted AUC passes sample_weight                                     #
# --------------------------------------------------------------------------- #
def test_weighted_auc_passes_sample_weight(run):
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    df = pd.read_parquet(M.PREDICTIONS_PARQUET)
    y = df["y"].to_numpy(np.uint8)
    w = df["w"].to_numpy(np.float64)
    p = df["p_a"].to_numpy(np.float64)
    exp_auc = float(roc_auc_score(y, p, sample_weight=w))
    summary_auc = run["summary"]["pooled"]["A"]["roc_auc"]
    assert abs(summary_auc - exp_auc) < 1e-12


# --------------------------------------------------------------------------- #
# FIX8.6 synthetic: opportunity-weighted (not raw-row) result returned        #
# --------------------------------------------------------------------------- #
def test_opportunity_weighted_not_raw_row():
    y = np.array([1, 1, 0, 0, 0], dtype=np.uint8)
    pred = np.array([1, 0, 0, 1, 1], dtype=np.uint8)
    # weights: trade1 (rows 0,1) sum=1 ; trade2 (rows 2,3,4) sum=1 ; but unequal per-row
    w = np.array([0.9, 0.1, 0.1, 0.5, 0.4], dtype=np.float64)
    rec = M._recall_metrics(pred, y, w)
    # opportunity-weighted LONG recall: row0 correct(1)->0.9, row1 wrong->0 ; trade sum=1
    # => long_recall = 0.9 / (0.9+0.1) = 0.9
    assert abs(rec["long_recall"] - 0.9) < 1e-12
    # raw-row (unweighted) long recall would be 0.5 -> must differ (proves weighting)
    raw = float(((pred == 1) & (y == 1)).sum()) / int((y == 1).sum())
    assert abs(rec["long_recall"] - raw) > 1e-9


# --------------------------------------------------------------------------- #
# FIX8.7 weighted classification metrics reproduce frozen methodology           #
# --------------------------------------------------------------------------- #
def test_classification_metrics_reproduce_frozen(run):
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    df = pd.read_parquet(M.PREDICTIONS_PARQUET)
    y = df["y"].to_numpy(np.uint8)
    w = df["w"].to_numpy(np.float64)
    for suf, key in (("a", "A"), ("b", "B"), ("c", "C")):
        pred = df[f"pred_{suf}"].to_numpy(np.uint8)
        p = df[f"p_{suf}"].to_numpy(np.float64)
        rec = M._recall_metrics(pred, y, w)
        acc = float(np.average(pred == y, weights=w))
        auc = float(roc_auc_score(y, p, sample_weight=w))
        pooled = run["summary"]["pooled"][key]
        assert abs(pooled["accuracy"] - acc) < 1e-12
        assert abs(pooled["long_recall"] - rec["long_recall"]) < 1e-12
        assert abs(pooled["short_recall"] - rec["short_recall"]) < 1e-12
        assert abs(pooled["predicted_long_share"] - rec["predicted_long_share"]) < 1e-12
        assert abs(pooled["roc_auc"] - auc) < 1e-12


# --------------------------------------------------------------------------- #
# FIX8.8 C evidence carries complementary_isomorphic_control marker             #
# --------------------------------------------------------------------------- #
def test_c_model_role_marker(run):
    assert run["summary"]["model_roles"]["C"] == "complementary_isomorphic_control"


# --------------------------------------------------------------------------- #
# FIX8.9 forbidden sentence must NOT appear in summary/evidence                #
# --------------------------------------------------------------------------- #
def test_no_forbidden_sentence(run):
    import json as _json
    s = _json.dumps(run["summary"])
    assert "Long/Short do not need different decision functions" not in s
    assert "do not need different decision functions" not in s
    # also scan the per-symbol + loso CSV evidence
    for p in (run["paths"]["per_symbol_csv"], run["paths"]["loso_csv"]):
        txt = open(p).read()
        assert "do not need different decision functions" not in txt


# --------------------------------------------------------------------------- #
# FIX8.10 B described as symmetry augmentation, not merely coordinate transform #
# --------------------------------------------------------------------------- #
def test_b_symmetry_augmentation_not_coordinate_only(run):
    assert "symmetry" in run["summary"]["model_roles"]["B"]
    assert "symmetry_augmentation" in run["summary"]["model_roles"]["B"]
    # interpretation must not reduce B to a pure coordinate transform
    assert "coordinate transform" not in run["summary"]["interpretation"]


# --------------------------------------------------------------------------- #
# FIX8.11 economic results and all paired deltas unchanged                     #
# --------------------------------------------------------------------------- #
def test_economic_results_unchanged(run):
    s = run["summary"]
    exp = M.FROZEN_ECONOMIC
    assert abs(s["pooled"]["A"]["return_atr"] - exp["A"]) < 1e-9
    assert abs(s["pooled"]["B"]["return_atr"] - exp["B"]) < 1e-9
    assert abs(s["pooled"]["C"]["return_atr"] - exp["C"]) < 1e-9
    pc = s["paired_contrasts"]
    for k in ("B_minus_A", "C_minus_B", "C_minus_A",
              "B_minus_A_LONG", "B_minus_A_SHORT",
              "C_minus_B_LONG", "C_minus_B_SHORT"):
        assert abs(pc[k]["mean"] - exp[k]) < 1e-9


# --------------------------------------------------------------------------- #
# FIX8.12 refresh path invokes NO training function                            #
# --------------------------------------------------------------------------- #
def test_refresh_path_no_training(run):
    from unittest import mock
    with mock.patch.object(M, "fit_direction_model") as m_fit, \
         mock.patch.object(M, "fit_a") as m_a, \
         mock.patch.object(M, "fit_b") as m_b, \
         mock.patch.object(M, "fit_c") as m_c:
        res = M.refresh_classification_metrics(save=False, verbose=False)
    assert m_fit.call_count == 0
    assert m_a.call_count == 0
    assert m_b.call_count == 0
    assert m_c.call_count == 0
    # and it still reproduces the weighted accuracy from the persisted predictions
    assert "A" in res and "accuracy" in res["A"]

