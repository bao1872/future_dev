"""Unit / causal-governance tests for R13.8-V2 factorial (plan §51-§56).

Uses a synthetic direction-conditioned frame for crossfit guarantees, and the
real canonical Direction dataset only where the crosswalk / A9==router identity
must be proven empirically.
"""
import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.r13_8_factorial_v2 import (
    ARCHS,
    HORIZON,
    PGM3,
    block_index_map,
    build_direction_assets,
    direction_conditioned_value,
    load_value_root,
    map_epochs,
    meta_state,
    pgm_crossfit,
    run_axis_chain,
    verify_crosswalk,
)
from research.liquidity_oracle_atlas.meta_output_integration_v1 import PGM3 as PGM3_R137
from research.liquidity_oracle_atlas import run_decomposed_v2_research as R

RNG = np.random.default_rng(20260926)


def _synthetic_conditioned(n_per_fold=40):
    rows = []
    base = pd.Timestamp("2025-01-01")
    for f in range(5):
        dt = base + pd.Timedelta(days=f * 10)
        for i in range(n_per_fold):
            sym = f"S{(i % 3)}"
            db = f * 1000 + i
            rows.append({
                "symbol": sym, "decision_bar": db, "side": "LONG",
                "fold": f,
                "decision_time": dt + pd.Timedelta(hours=i % 5),
                "label_available_time": dt + pd.Timedelta(days=1),
                "trading_day": dt.date(),
                "epoch_weight": 1.0,
                "episode_return_atr": float(RNG.normal(0, 0.5)),
                "p_win": float(RNG.uniform(0.2, 0.8)),
                "mu_win": float(RNG.uniform(0.5, 3.0)),
                "mu_loss": float(RNG.uniform(0.5, 3.0)),
            })
    return pd.DataFrame(rows)


def test_block_index_complete_5day():
    days = np.array([f"2025-01-{d:02d}" for d in range(1, 13)] +
                   [f"2025-02-{d:02d}" for d in range(1, 4)], dtype="datetime64")
    idx = block_index_map(days, 5)
    assert len(idx) == 3  # 15 days -> 3 complete blocks
    assert all(len(b) == 5 for b in idx)


def test_pgm_crossfit_fold0_warmup_and_history_causality():
    frame = _synthetic_conditioned()
    res, rp, meta = pgm_crossfit(frame)
    # fold0 excluded from evaluation
    assert (res["fold"] == 0).sum() == 0
    assert set(res["fold"].unique()) <= {1, 2, 3, 4}
    # PGM is reused R13.7 class (K=3)
    assert isinstance(PGM3(), PGM3_R137)
    # q80 thresholds exist per eval fold
    for g in ["P", "EV_C", "PGM3"]:
        assert (f"q80_{g}" in res.columns)
    # selection fraction is roughly the causal 80th percentile (select top 20%)
    for g in ["P", "EV_C", "PGM3"]:
        frac = res[f"select20_{g}"].mean()
        assert 0.10 < frac < 0.30


def test_pgm_only_sees_three_outputs():
    df = _synthetic_conditioned()
    z = meta_state(df)
    assert z.shape == (len(df), 3)
    p = np.clip(df["p_win"].to_numpy(float), 1e-6, 1 - 1e-6)
    assert np.allclose(z[:, 0], np.log(p / (1 - p)))
    assert np.allclose(z[:, 1], np.log1p(np.maximum(df["mu_win"], 0)))
    assert np.allclose(z[:, 2], np.log1p(np.maximum(df["mu_loss"], 0)))


def test_crosswalk_one_to_one():
    split, ds, data, dir_index = build_direction_assets()
    vr = load_value_root(ARCHS["A0"])
    ep = map_epochs(vr, dir_index)
    xw = verify_crosswalk(ep, dir_index)
    assert xw["n_missing"] == 0
    assert xw["n_duplicate_value_keys"] == 0
    assert xw["n_duplicate_direction_keys"] == 0
    assert xw["n_value_epochs"] == xw["n_direction_matches"]


def test_a9_equals_frozen_router():
    """A9 hard output must equal the canonical frozen router (chain['router_te'])."""
    split, ds, data, dir_index = build_direction_assets()
    vr = load_value_root(ARCHS["A0"])
    ep = map_epochs(vr, dir_index)
    target = ep[ep["fold"] == 0]
    target_rows = target["dir_row"].to_numpy(int)
    T_ns = int(pd.to_datetime(target["decision_time"]).min().value)
    cand = pd.to_datetime(ds["candidate_decision_time"]).to_numpy("datetime64[ns]").astype("int64")
    oexit = pd.to_datetime(ds["oracle_exit_fill_time"]).to_numpy("datetime64[ns]").astype("int64")
    avail = np.where((cand < T_ns) & (oexit < T_ns))[0]
    fit_idx, val_idx = _split_85_15_test(data, avail)
    chain = run_axis_chain(data, ds, fit_idx, val_idx, target_rows)
    a9 = np.asarray(chain["A"]).astype(np.uint8)
    router_te = np.asarray(chain["router_te"]).astype(np.uint8)
    np.testing.assert_array_equal(a9, router_te)


def _split_85_15_test(data, available_idx):
    # local copy of the module helper for the single-fold test
    from research.liquidity_oracle_atlas.r13_8_factorial_v2 import split_85_15
    return split_85_15(data, available_idx)


def test_no_m9_m33_e33_in_axis_chain():
    """run_axis_chain must not fit M9/M33/E33; only router A + E9 gated experts."""
    split, ds, data, dir_index = build_direction_assets()
    vr = load_value_root(ARCHS["A0"])
    ep = map_epochs(vr, dir_index)
    target = ep[ep["fold"] == 0]
    target_rows = target["dir_row"].to_numpy(int)
    T_ns = int(pd.to_datetime(target["decision_time"]).min().value)
    cand = pd.to_datetime(ds["candidate_decision_time"]).to_numpy("datetime64[ns]").astype("int64")
    oexit = pd.to_datetime(ds["oracle_exit_fill_time"]).to_numpy("datetime64[ns]").astype("int64")
    avail = np.where((cand < T_ns) & (oexit < T_ns))[0]
    fit_idx, val_idx = _split_85_15_test(data, avail)
    chain = run_axis_chain(data, ds, fit_idx, val_idx, target_rows)
    # E9 gated experts present; M9/M33/E33 are simply absent from the returned dict
    assert "E9" in chain and "A" in chain
    assert "m9" not in chain["models"] and "m33" not in chain["models"]


def test_direction_epoch_weight_is_one():
    """direction_conditioned_value assigns exactly one side per epoch, weight=1."""
    split, ds, data, dir_index = build_direction_assets()
    vr = load_value_root(ARCHS["A0"])
    ep = map_epochs(vr, dir_index)
    axis, _ = _build_axis(split, ds, data, dir_index, ep)
    cond = direction_conditioned_value(vr, axis, "A9")
    assert (cond["epoch_weight"] == 1.0).all()
    assert not cond.duplicated(["symbol", "decision_bar"]).any()


def _build_axis(split, ds, data, dir_index, ep):
    from research.liquidity_oracle_atlas.r13_8_factorial_v2 import build_direction_axis
    return build_direction_axis(ds, data, dir_index, ep)
