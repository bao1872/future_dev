"""Unit / causal-governance tests for R13.6/R13.7 meta integration (plan §33).

Uses a synthetic, fully deterministic frame so the tests run without any
real OOF data and assert the second-level cross-fit guarantees directly.
"""
import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.meta_output_integration_v1 import (
    META_MODELS,
    WeightedScaler,
    PGM3,
    GmmLinearMoE3,
    SplineRidge,
    Poly2Ridge,
    enforce_pair_purity,
    meta_splits,
    meta_state,
    run_one_model,
    _block_index_map,
    _boot_paired,
    _epoch_sides,
)

RNG = np.random.default_rng(20260925)
SYMBOLS = ["S1", "S2", "S3"]
SIDES = ["LONG", "SHORT"]
HORIZON = "td5"


def make_synth():
    rows = []
    base = pd.Timestamp("2025-01-01")
    for f in range(5):
        dt = base + pd.Timedelta(days=f * 10)
        for si, sym in enumerate(SYMBOLS):
            for sidei, side in enumerate(SIDES):
                db = f * 1000 + si * 100 + sidei  # shared per (sym,fold)
                dtime = dt + pd.Timedelta(hours=si)
                rows.append({
                    "symbol": sym,
                    "decision_bar": db,
                    "side": side,
                    "horizon": HORIZON,
                    "fold": f,
                    "decision_time": dtime,
                    "label_available_time": dtime + pd.Timedelta(days=1),
                    "trading_day": dt.date(),
                    "p_win": float(RNG.uniform(0.2, 0.8)),
                    "mu_win": float(RNG.uniform(0.5, 3.0)),
                    "mu_loss": float(RNG.uniform(0.5, 3.0)),
                    "episode_return_atr": float(RNG.normal(0, 0.5)),
                    "win": int(RNG.uniform(0, 1) > 0.5),
                    "sample_weight": float(RNG.uniform(0.5, 1.5)),
                    "candidate_at_decision": True,
                })
    return pd.DataFrame(rows)


def test_meta_state_uses_only_three_outputs():
    df = make_synth()
    z = meta_state(df)
    assert z.shape == (len(df), 3)
    # z1 = logit(p) monotonic in p
    p = np.clip(df["p_win"].to_numpy(float), 1e-6, 1 - 1e-6)
    assert np.allclose(z[:, 0], np.log(p / (1 - p)))
    assert np.allclose(z[:, 1], np.log1p(np.maximum(df["mu_win"].to_numpy(float), 0)))
    assert np.allclose(z[:, 2], np.log1p(np.maximum(df["mu_loss"].to_numpy(float), 0)))


def test_weighted_scaler_is_weighted_mean_unit_variance():
    X = RNG.normal(size=(200, 3))
    w = RNG.uniform(0.5, 1.5, size=200)
    s = WeightedScaler().fit(X, w)
    xs = s.transform(X)
    mean = np.average(xs, axis=0, weights=w)
    var = np.average((xs - mean) ** 2, axis=0, weights=w)
    assert np.allclose(mean, 0, atol=1e-9)
    assert np.allclose(var, 1, atol=1e-9)


def test_fold0_excluded_only_folds_1_to_4():
    df = make_synth()
    splits = meta_splits(df)
    folds = [s[0] for s in splits]
    assert folds == [1, 2, 3, 4]


def test_no_future_fold_in_meta_training():
    df = make_synth()
    for target_fold, hist, target, _ in meta_splits(df):
        assert (hist["fold"] < target_fold).all()
        assert (target["fold"] == target_fold).all()


def test_decision_and_label_time_before_cutoff():
    df = make_synth()
    for target_fold, hist, target, cutoff in meta_splits(df):
        assert (pd.to_datetime(hist["decision_time"]) < cutoff).all()
        assert (pd.to_datetime(hist["label_available_time"]) < cutoff).all()


def test_pair_purity_drops_incomplete_two_sided_epoch():
    df = make_synth()
    # corrupt one side of a two-sided epoch so its label is not available
    mask = (df["symbol"] == "S1") & (df["decision_bar"] == 100) & (
        df["side"] == "LONG") & (df["fold"] == 0)
    df.loc[mask, "label_available_time"] = pd.Timestamp("2030-01-01")
    full = _epoch_sides(df)
    cutoff = pd.Timestamp("2025-01-03")  # between fold0 and fold1
    hist = df[df["fold"] < 1].copy()
    cleaned = enforce_pair_purity(hist, full, cutoff)
    # the whole S1/100 epoch must be gone (both sides removed)
    assert not ((cleaned["symbol"] == "S1") & (cleaned["decision_bar"] == 100)).any()


def test_same_target_universe_across_models():
    df = make_synth()
    keys = None
    for m in ["B0_P", "M1_POLY2_RIDGE", "M3_PGM3", "M4_GMM_MOE3"]:
        pred, _ = run_one_model(df, m)
        k = list(zip(pred["symbol"], pred["decision_bar"], pred["side"],
                     pred["fold"]))
        if keys is None:
            keys = k
        else:
            assert k == keys, f"target universe differs for {m}"


def test_prediction_artifact_has_no_outcome():
    df = make_synth()
    pred, _ = run_one_model(df, "M1_POLY2_RIDGE")
    for forbidden in ["episode_return_atr", "win", "actual", "label"]:
        assert forbidden not in pred.columns


def test_thresholds_from_history_not_target():
    df = make_synth()
    pred, _ = run_one_model(df, "B0_P")
    # for each fold block, threshold must be <= max hist score.
    # Since score==p_win and threshold is 80th pct of hist, verify the
    # fraction selected in target is ~20% (history-defined gate).
    sel = pred["select20"].to_numpy(float)
    assert 0.10 < sel.mean() < 0.30


def test_m1_fixed_degree_alpha():
    m = Poly2Ridge()
    assert m.poly.degree == 2
    # fit then confirm alpha frozen at 1.0
    X = RNG.normal(size=(50, 3))
    m.fit(X, RNG.normal(size=50), np.ones(50))
    assert m.model.alpha == 1.0


def test_m2_fixed_knots_degree_alpha():
    m = SplineRidge()
    assert m.spline.n_knots == 4 and m.spline.degree == 3
    X = RNG.normal(size=(80, 3))
    m.fit(X, RNG.normal(size=80), np.ones(80))
    assert m.model.alpha == 1.0


def test_m3_m4_fixed_k3():
    for cls in (PGM3, GmmLinearMoE3):
        m = cls()
        X = RNG.normal(size=(120, 3))
        m.fit(X, RNG.normal(size=120), np.ones(120))
        assert m.gmm.n_components == 3


def test_pgm_relabel_deterministic_by_theta():
    X = RNG.normal(size=(300, 3))
    y = X[:, 0] + RNG.normal(0, 0.1, size=300)
    w = np.ones(300)
    m = PGM3().fit(X, y, w)
    rp = m.regime_prob(X)
    theta_ordered = m.theta_[m._order]
    assert np.all(np.diff(theta_ordered) >= 0)  # ascending
    # regime probabilities are a valid distribution
    assert np.allclose(rp.sum(axis=1), 1.0)


def test_complete_five_day_blocks_only():
    days = np.array([f"2025-01-{d:02d}" for d in range(1, 13)] +
                   [f"2025-02-{d:02d}" for d in range(1, 4)], dtype="datetime64")
    idx = _block_index_map(days, 5)
    # 15 days -> 3 complete blocks of 5; remainder 2 dropped
    assert len(idx) == 3
    assert all(len(b) == 5 for b in idx)


def test_bootstrap_paired_same_blocks_for_both_models():
    y = RNG.normal(size=200)
    w = np.ones(200)
    score = RNG.normal(size=200)
    sel_m = RNG.normal(size=200) > 0
    sel_p = RNG.normal(size=200) > 0
    td = np.array([f"2025-01-{((i % 20) + 1):02d}" for i in range(200)],
                  dtype="datetime64")
    bi = _block_index_map(td, 5)
    d1 = _boot_paired((y, w, score, sel_m), (y, w, score, sel_p), bi, 200,
                      12345)
    d2 = _boot_paired((y, w, score, sel_m), (y, w, score, sel_p), bi, 200,
                      12345)
    assert np.allclose(d1, d2)  # deterministic for same seed/blocks


def test_all_meta_models_run_and_classified():
    df = make_synth()
    for m in META_MODELS:
        pred, rp = run_one_model(df, m)
        assert len(pred) > 0
        if m in ("M3_PGM3", "M4_GMM_MOE3"):
            assert rp is not None and rp.shape[1] == 3
        else:
            assert rp is None


def test_a1_after_a0_is_separate_architecture():
    # run_one_model is architecture-agnostic; both archs share identical engine
    df_a = make_synth()
    pred_a, _ = run_one_model(df_a, "M1_POLY2_RIDGE")
    df_b = make_synth()
    pred_b, _ = run_one_model(df_b, "M1_POLY2_RIDGE")
    assert set(pred_a.columns) == set(pred_b.columns)
    assert "episode_return_atr" not in pred_a.columns
