"""Unit / causal-governance tests for R13.8-V2 factorial (plan §51-§56).

Uses a synthetic direction-conditioned frame for crossfit guarantees, and the
real canonical Direction dataset only where the crosswalk / A9==router identity
must be proven empirically.
"""
import os

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.r13_8_factorial_v2 import (
    ARCHS,
    DIR_SYSTEMS,
    EVIDENCE_DIR,
    FACT_CONTRAST_CSV,
    FACTORIAL_OOF_PARQUET,
    GATES,
    HORIZON,
    MATCHEDN_CSV,
    N_FOLDS,
    PGM3,
    DIRECTION_AXIS_PARQUET,
    block_index_map,
    build_direction_assets,
    direction_conditioned_value,
    direction_history_pool,
    epoch_universe_identity,
    load_value_root,
    map_epochs,
    meta_state,
    pair_universe_audit,
    pgm_crossfit,
    _wmean,
    observed_matched_return,
    boot_matched_pair,
    corrected_fit_counts,
    router_probability_gate,
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


# --------------------------------------------------------------------------- #
# Reviewer repair §2-§6 / §13: evidence & governance regression tests           #
# --------------------------------------------------------------------------- #
_REPAIR_CACHE = {}


def _repair_inputs():
    """Load the frozen prediction artifacts + value roots once per test session."""
    if not _REPAIR_CACHE:
        split, ds, data, dir_index = build_direction_assets()
        value_roots = {a: load_value_root(arch) for a, arch in ARCHS.items()}
        axis = pd.read_parquet(DIRECTION_AXIS_PARQUET)
        fact_ledger = pd.read_parquet(FACTORIAL_OOF_PARQUET)
        _REPAIR_CACHE.update(split=split, ds=ds, data=data, dir_index=dir_index,
                             value_roots=value_roots, axis=axis, fact_ledger=fact_ledger)
    return _REPAIR_CACHE


def test_pair_complete_long_short_per_epoch():
    """§2 / §13.1: every (symbol,decision_bar,fold) has exactly LONG & SHORT."""
    inp = _repair_inputs()
    for a in ARCHS:
        vr = inp["value_roots"][a]
        for _, g in vr.groupby(["symbol", "decision_bar", "fold"]):
            assert len(g) == 2
            assert set(g["side"].tolist()) == {"LONG", "SHORT"}


def test_pair_sample_weight_sums_to_one():
    """§2 / §13.2: pair sample weights sum to 1.0 per epoch."""
    inp = _repair_inputs()
    for a in ARCHS:
        vr = inp["value_roots"][a]
        for _, g in vr.groupby(["symbol", "decision_bar", "fold"]):
            assert abs(float(g["sample_weight"].sum()) - 1.0) <= 1e-12


def test_pair_universe_audit_helper_passes():
    """§2 / §13.1-2: the audit helper itself asserts completeness + weight drift."""
    inp = _repair_inputs()
    for a in ARCHS:
        out = pair_universe_audit(inp["value_roots"][a])
        assert out["incomplete_epochs"] == 0
        assert out["max_pair_weight_error"] <= 1e-12


def test_a0_a1_epoch_universe_identical():
    """§3 / §13.3: A0 and A1 share the exact same epoch universe."""
    inp = _repair_inputs()
    a0 = inp["value_roots"]["A0"]
    a1 = inp["value_roots"]["A1"]
    keys = ["symbol", "decision_bar", "fold", "decision_time", "trading_day"]
    s0 = set(map(tuple, a0[keys].drop_duplicates().to_numpy()))
    s1 = set(map(tuple, a1[keys].drop_duplicates().to_numpy()))
    assert s0 == s1
    # explicit helper cross-check
    ident = epoch_universe_identity(a0, a1)
    assert ident["identical"] is True


def test_direction_selection_one_side_per_epoch():
    """§3 / §13.4: each Direction chooses exactly one side for all 21341 epochs."""
    inp = _repair_inputs()
    for D in DIR_SYSTEMS:
        for a in ARCHS:
            chosen = direction_conditioned_value(inp["value_roots"][a], inp["axis"], D)
            assert len(chosen) == 21341
            assert not chosen.duplicated(["symbol", "decision_bar"]).any()


def test_direction_axis_probability_reproduction():
    """§4 / §13.5: fixed-router p_long vs A9 m0 p_long within 1e-4 per fold."""
    inp = _repair_inputs()
    gate = router_probability_gate(inp["axis"])
    for k in range(N_FOLDS):
        dev = float(gate[gate["fold"] == k]["max_abs_a9_minus_router_p"].iloc[0])
        assert dev <= 1e-4


def test_direction_history_frozen_train_only():
    """§5 / §13.6: Direction FIT/VAL pool is a subset of the frozen TRAIN index,
    hence disjoint from the old VAL/TEST indices."""
    inp = _repair_inputs()
    split = inp["split"]; ds = inp["ds"]; data = inp["data"]
    train_idx = np.asarray(split["train_idx"])
    val_idx = np.asarray(split["val_idx"])
    test_idx = np.asarray(split["test_idx"])
    # frozen split invariant: TRAIN disjoint from VAL/TEST
    assert len(set(train_idx.tolist()) & set(val_idx.tolist())) == 0
    assert len(set(train_idx.tolist()) & set(test_idx.tolist())) == 0
    axis = inp["axis"]
    for k in range(N_FOLDS):
        target = axis[axis["fold"] == k]
        T_ns = int(pd.to_datetime(target["decision_time"]).min().value)
        fit_idx, val2 = direction_history_pool(data, ds, T_ns, train_idx)
        fit_val = set(fit_idx.tolist()) | set(val2.tolist())
        # Direction FIT and VAL are subsets of TRAIN -> disjoint from old VAL/TEST
        assert fit_val <= set(train_idx.tolist())
        assert len(fit_val & set(val_idx.tolist())) == 0
        assert len(fit_val & set(test_idx.tolist())) == 0


def test_matchedN_uses_p_baseline_and_pgm_challenger():
    """§7 / §13.7-8: matched-N compares top-N by P vs top-N by PGM3 (EV_C excluded).

    matched20 point = top-ceil(0.2n)-by-SCORE (deterministic), not the causal-q80
    selected-mean. matchedN point = top-kP-by-SCORE where kP = causal-q80 P count.
    """
    m = pd.read_csv(str(MATCHEDN_CSV))
    # EV_C / gate axis never enters the matched-N P-vs-PGM evidence
    assert set(m["direction"].unique()) == {"A9", "E9"}
    assert set(m["value_arch"].unique()) == {"A0", "A1"}
    assert "EV_C" not in m.columns
    r20 = m[(m.direction == "A9") & (m.value_arch == "A0") & (m.match_kind == "matched20")].iloc[0]
    rN = m[(m.direction == "A9") & (m.value_arch == "A0") & (m.match_kind == "matchedN")].iloc[0]
    # matched20: top-20%-by-score (deterministic)
    assert abs(r20["P_return"] - 0.170042) < 1e-4
    assert abs(r20["PGM_return"] - 0.061070) < 1e-4
    assert abs(r20["PGM_minus_P_point"] - (0.061070 - 0.170042)) < 1e-4
    # matchedN: top-kP-by-score (kP = causal-q80 P count)
    assert abs(rN["P_return"] - 0.150116) < 1e-4
    assert abs(rN["PGM_return"] - 0.058482) < 1e-4
    assert abs(rN["PGM_minus_P_point"] - (0.058482 - 0.150116)) < 1e-4
    # P and PGM3 are distinct (the original bug wrote PGM into all rows)
    assert abs(r20["P_return"] - r20["PGM_return"]) > 1e-3


def test_contrasts_a0_and_a1_five_rows_each():
    """§8 / §13.9: both A0 and A1 produce all 5 pre-registered contrasts."""
    c = pd.read_csv(str(FACT_CONTRAST_CSV))
    assert len(c[c.value_arch == "A0"]) == 5
    assert len(c[c.value_arch == "A1"]) == 5
    expected = {"PGM_minus_P_given_A9", "PGM_minus_P_given_E9",
                "E9_minus_A9_given_P", "E9_minus_A9_given_PGM",
                "direction_gate_interaction"}
    assert set(c[c.value_arch == "A0"]["contrast"]) == expected
    assert set(c[c.value_arch == "A1"]["contrast"]) == expected


def test_contrast_point_is_observed_difference():
    """§9 / §13.10: contrast `point` is the observed-data contrast, not the
    bootstrap mean."""
    c = pd.read_csv(str(FACT_CONTRAST_CSV))
    a0 = c[c.value_arch == "A0"].set_index("contrast")
    assert abs(a0.loc["PGM_minus_P_given_A9", "point"] - (0.068888 - 0.150116)) < 1e-4
    a1 = c[c.value_arch == "A1"].set_index("contrast")
    assert abs(a1.loc["PGM_minus_P_given_E9", "point"] - 0.134249) < 1e-4


def test_direction_fit_count_includes_prequential_and_experts():
    """§6 / §13.11: underlying fit count = 1 m0 + 1 fixed router + N prequential
    OOF router + 2 E9 experts = 4 + N (experts counted as two, not one)."""
    p = pd.read_csv(os.path.join(EVIDENCE_DIR, "r13_8_direction_provenance_v2.csv"))
    assert (p["n_underlying_model_fits"] == 4 + p["n_prequential_oof_router_fits"]).all()
    assert p["n_underlying_model_fits"].sum() == 29
    assert (p["n_underlying_model_fits"] >= 4).all()


# --------------------------------------------------------------------------- #
# Final matched-coverage patch (§1-§7): observed_matched_return + joint mask     #
# --------------------------------------------------------------------------- #
def _build_cell(D, Va, G):
    """Rebuild a (score, y, w, fold, sel) cell from the frozen fact_ledger +
    value roots exactly as main() does."""
    inp = _repair_inputs()
    led = inp["fact_ledger"][
        (inp["fact_ledger"].direction == D)
        & (inp["fact_ledger"].value_arch == Va)
        & (inp["fact_ledger"].gate == G)].copy()
    chosen = direction_conditioned_value(inp["value_roots"][Va], inp["axis"], D)
    ch = chosen[["symbol", "decision_bar", "fold", "episode_return_atr"]].copy()
    merged = led.merge(ch, on=["symbol", "decision_bar", "fold"], how="left")
    return (merged["score"].to_numpy(float),
            merged["episode_return_atr"].to_numpy(float),
            merged["epoch_weight"].to_numpy(float),
            merged["fold"].to_numpy(),
            merged["trade20"].to_numpy(bool))


def test_matched20_point_equals_topk_by_score():
    """§1-§3 / §6.1: matched20 point equals the direct deterministic top-20%-by-score
    return (recomputed from the frozen ledger)."""
    scP, yP, wP, foldP, _ = _build_cell("A9", "A0", "P")
    scG, yG, wG, foldG, _ = _build_cell("A9", "A0", "PGM3")
    folds = np.unique(foldP)
    k20 = {int(f): max(1, int(np.ceil(0.20 * (foldP == f).sum()))) for f in folds}
    P20 = observed_matched_return(scP, yP, wP, foldP, k20)
    PGM20 = observed_matched_return(scG, yG, wG, foldG, k20)
    m = pd.read_csv(str(MATCHEDN_CSV))
    row = m[(m.direction == "A9") & (m.value_arch == "A0") & (m.match_kind == "matched20")].iloc[0]
    assert abs(row["P_return"] - P20) < 1e-6
    assert abs(row["PGM_return"] - PGM20) < 1e-6


def test_matched20_point_not_causal_q80_mean():
    """§2 / §6.2: matched20 point must NOT be the causal-q80 selected-mean return
    (unless coincidentally identical)."""
    scP, yP, wP, foldP, selP = _build_cell("A9", "A0", "P")
    q80_mean = _wmean(yP[selP], wP[selP])
    m = pd.read_csv(str(MATCHEDN_CSV))
    row = m[(m.direction == "A9") & (m.value_arch == "A0") & (m.match_kind == "matched20")].iloc[0]
    assert abs(row["P_return"] - q80_mean) > 1e-4


def test_matchedN_point_equals_same_N_selection():
    """§3 / §6.3: matchedN point equals the direct same-N (kP) top-by-score return,
    where kP = causal-q80 P selection count per fold."""
    scP, yP, wP, foldP, selP = _build_cell("A9", "A0", "P")
    scG, yG, wG, foldG, selG = _build_cell("A9", "A0", "PGM3")
    folds = np.unique(foldP)
    kP = {int(f): int(((foldP == f) & selP).sum()) for f in folds}
    PN = observed_matched_return(scP, yP, wP, foldP, kP)
    PGMN = observed_matched_return(scG, yG, wG, foldG, kP)
    m = pd.read_csv(str(MATCHEDN_CSV))
    row = m[(m.direction == "A9") & (m.value_arch == "A0") & (m.match_kind == "matchedN")].iloc[0]
    assert abs(row["P_return"] - PN) < 1e-6
    assert abs(row["PGM_return"] - PGMN) < 1e-6


def test_matched20_n_per_fold_equals_k20():
    """§4 / §6.4: matched20 n_per_fold encodes ceil(0.20 * per-fold count)."""
    scP, yP, wP, foldP, _ = _build_cell("A9", "A0", "P")
    folds = np.unique(foldP)
    k20 = {int(f): max(1, int(np.ceil(0.20 * (foldP == f).sum()))) for f in folds}
    m = pd.read_csv(str(MATCHEDN_CSV))
    row = m[(m.direction == "A9") & (m.value_arch == "A0") & (m.match_kind == "matched20")].iloc[0]
    n_per = {int(k): int(v) for k, v in (p.split("=") for p in row["n_per_fold"].split(","))}
    assert n_per == k20


def test_matchedN_n_per_fold_equals_kP():
    """§5 / §6.5: matchedN n_per_fold encodes the causal-q80 P selection count."""
    scP, yP, wP, foldP, selP = _build_cell("A9", "A0", "P")
    folds = np.unique(foldP)
    kP = {int(f): int(((foldP == f) & selP).sum()) for f in folds}
    m = pd.read_csv(str(MATCHEDN_CSV))
    row = m[(m.direction == "A9") & (m.value_arch == "A0") & (m.match_kind == "matchedN")].iloc[0]
    n_per = {int(k): int(v) for k, v in (p.split("=") for p in row["n_per_fold"].split(","))}
    assert n_per == kP


def test_bootstrap_paired_alignment():
    """§4 / §6.6: boot_matched_pair returns P/PGM arrays of equal length and paired
    replicate alignment; with identical scores p_ret == g_ret exactly."""
    n = 400
    fold = np.repeat(np.arange(1, 5), 100)
    rng = np.random.default_rng(0)
    score = rng.random(n)
    y = rng.standard_normal(n)
    w = np.ones(n)
    bi = block_index_map(np.arange(n), 5)
    p_ret, g_ret = boot_matched_pair(score, score, y, w, fold, bi,
                                     {f: 10 for f in range(1, 5)}, 200, 12345)
    assert p_ret.shape == g_ret.shape
    assert np.isfinite(p_ret).sum() == np.isfinite(g_ret).sum()
    assert np.allclose(p_ret, g_ret, equal_nan=True)


def test_corrected_fit_counts_use_frozen_train():
    """§5 / §6.7: corrected fit counts are derived from the frozen-TRAIN-pool
    direction_history_pool and remain 5/5/6/6/7 (total 29)."""
    inp = _repair_inputs()
    split = inp["split"]; ds = inp["ds"]; data = inp["data"]
    train_idx = np.asarray(split["train_idx"])
    fc = corrected_fit_counts(inp["axis"], ds, data, train_idx)
    assert list(fc["n_underlying_model_fits"]) == [5, 5, 6, 6, 7]
    assert int(fc["n_underlying_model_fits"].sum()) == 29
    for k in range(N_FOLDS):
        target = inp["axis"][inp["axis"]["fold"] == k]
        T_ns = int(pd.to_datetime(target["decision_time"]).min().value)
        fit_idx, val_idx = direction_history_pool(data, ds, T_ns, train_idx)
        assert set(fit_idx.tolist()) | set(val_idx.tolist()) <= set(train_idx.tolist())
