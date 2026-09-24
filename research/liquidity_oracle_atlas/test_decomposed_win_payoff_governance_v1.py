"""Governance / efficiency regression tests for the decomposed WIN/PAYOFF v1
pipeline (reviewer verdict NEED REVISION — GOVERNANCE / EFFICIENCY ONLY).

These cover the reviewer-required regressions that are cleanly unit-testable:

  G1  -> validated by the integration R8 materialization (state_v1.parquet must
         carry STRUCT33 + sup/res strength); see run at FINAL_CODE_SHA.
  G2  -> decomposed builder derives features ONLY from the state parquet; no
         environment reload / geometry extraction / load_symbol_state (#2,#3,#4,#5).
  G6/G7 -> WIN33 / PAY8 values equal the canonical side-oriented STRUCT33 /
         direct geometry references on sampled bars (#6,#7).
  G11 -> TEST prediction mask has a COMMON_END upper bound (#12).
  G12 -> missing R8 artifact hard-fails the formal runner (#13).
  G17 -> per-symbol point estimates use complete inference days only (#17).
  G21 -> Formal artifacts redirect into art_root and never touch canonical
         paths when mocked (#15).

The remaining gates (G13/G14/G19/G20) are exercised by the real formal run at
FINAL_CODE_SHA, not by unit tests.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

import research.liquidity_oracle_atlas.build_struct33_dataset_v1 as S33
import research.liquidity_oracle_atlas.structural_renewal_dataset_v1 as R8
import research.liquidity_oracle_atlas.build_decomposed_value_dataset_v1 as D
import research.liquidity_oracle_atlas.win_probability_model_v1 as R9A
import research.liquidity_oracle_atlas.payoff_ratio_model_v1 as R9B
import research.liquidity_oracle_atlas.sequential_decomposed_policy_v1 as R10


# --------------------------------------------------------------------------- #
# G2 / G6 / G7 : state-parquet builders (zero recompute, exact references)     #
# --------------------------------------------------------------------------- #
def _synthetic_state_df(n: int, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {}
    for c in S33.STRUCT33:
        cols[c] = rng.standard_normal(n).astype("float32")
    cols["sup_top"] = rng.random(n)
    cols["sup_bottom"] = rng.random(n)
    cols["sup_strength"] = rng.random(n)
    cols["res_top"] = rng.random(n)
    cols["res_bottom"] = rng.random(n)
    cols["res_strength"] = rng.random(n)
    cols["atr"] = rng.random(n) + 0.5
    cols["close"] = rng.random(n) + 10.0
    return pd.DataFrame(cols)


def test_decomposed_builder_has_no_env_dependency(monkeypatch):
    # G2 #2/#3: the decomposed module must not import / call load_symbol_state or
    # run_environment_m15 during feature construction.
    assert not hasattr(D, "load_symbol_state"), "load_symbol_state must not be imported"

    def _boom(*a, **k):
        raise RuntimeError("environment was (re)loaded")
    monkeypatch.setattr(R8, "run_environment_m15", _boom)
    sdf = _synthetic_state_df(8)
    # These must succeed even though run_environment_m15 now raises.
    df_w = D.build_win33_from_state(sdf, "X")
    df_p = D.build_pay8_from_state(sdf, "X")
    assert len(df_w) == 16 and len(df_p) == 16


def test_build_win33_from_state_matches_reference():
    # G6: WIN33 must equal the canonical side-oriented STRUCT33 reference.
    n = 9
    sdf = _synthetic_state_df(n)
    df = D.build_win33_from_state(sdf, "X")
    x33 = sdf[list(S33.STRUCT33)].to_numpy(np.float32)
    x33 = np.concatenate([x33, x33], axis=0)
    is_long = np.concatenate([np.ones(n, bool), np.zeros(n, bool)])
    ref = R8.orient_struct33_router_side(x33, is_long)
    got = df[list(D.WIN33_COLS)].to_numpy(np.float32)
    assert got.shape == (2 * n, len(D.WIN33_COLS))
    assert np.allclose(got, ref)


def test_build_pay8_from_state_matches_reference():
    # G7: PAY8 must equal the direct geometry reference.
    n = 9
    sdf = _synthetic_state_df(n)
    df = D.build_pay8_from_state(sdf, "X")
    b = np.tile(np.arange(n), 2)
    is_long = np.concatenate([np.ones(n, bool), np.zeros(n, bool)])
    sup_top = sdf["sup_top"].to_numpy(float)[b]
    sup_bottom = sdf["sup_bottom"].to_numpy(float)[b]
    sup_strength = sdf["sup_strength"].to_numpy(float)[b]
    res_top = sdf["res_top"].to_numpy(float)[b]
    res_bottom = sdf["res_bottom"].to_numpy(float)[b]
    res_strength = sdf["res_strength"].to_numpy(float)[b]
    atr = sdf["atr"].to_numpy(float)[b]
    close = sdf["close"].to_numpy(float)[b]
    side = np.where(is_long, 1.0, -1.0)
    fav, adv = R8.structural_barriers(is_long, sup_top, res_bottom)
    g, l, _elig = R8.bracket_metrics(side, close, fav, adv, atr)
    log_rr = R8._log_rr(g, l)
    ahead_width = np.where(is_long, (res_top - res_bottom) / atr,
                           (sup_top - sup_bottom) / atr)
    back_width = np.where(is_long, (sup_top - sup_bottom) / atr,
                          (res_top - res_bottom) / atr)
    ahead_strength = np.where(is_long, res_strength, sup_strength)
    back_strength = np.where(is_long, sup_strength, res_strength)
    atr_price = np.where(np.abs(close) > 0, atr / np.abs(close), np.nan)
    ref = np.column_stack([g, l, log_rr, ahead_width, back_width,
                           ahead_strength, back_strength, atr_price]).astype(np.float32)
    got = df[list(D.PAY8_COLS)].to_numpy(np.float32)
    assert got.shape == (2 * n, len(D.PAY8_COLS))
    assert np.allclose(got, ref, equal_nan=True)


def test_downstream_zero_recompute_counters():
    # G2 #4/#5: building from state parquet must NOT bump any env/geometry reload.
    sdf = _synthetic_state_df(6)
    D.reset_counters()
    D.build_win33_from_state(sdf, "X")
    D.build_pay8_from_state(sdf, "X")
    assert D.COUNTERS["environment_reloads_after_r8"] == 0
    assert D.COUNTERS["geometry_reloads_after_r8"] == 0
    assert D.COUNTERS["load_symbol_state_calls_after_r8"] == 0


# --------------------------------------------------------------------------- #
# G11 : closed TEST window (COMMON_END upper bound)                            #
# --------------------------------------------------------------------------- #
def _split_with_end(end: str):
    return {"cal": {"cuts": [pd.Timestamp("2020-01-01"),
                             pd.Timestamp("2020-01-03")],
                     "end": end}}


def test_test_window_mask_upper_bound_win():
    times = pd.date_range("2020-01-01", periods=10, freq="D")
    state = pd.DataFrame({"bar_index": np.arange(10),
                          "decision_time": times})
    split = _split_with_end("2020-01-08")
    m = R9A.test_window_mask(state, split)
    assert bool(m[2]) is True     # 2020-01-03 == T2 -> in
    assert bool(m[7]) is True     # 2020-01-08 == COMMON_END -> in
    assert bool(m[8]) is False    # 2020-01-09 > COMMON_END -> out
    assert bool(m[9]) is False    # 2020-01-10 > COMMON_END -> out


def test_test_window_mask_upper_bound_payoff():
    times = pd.date_range("2020-01-01", periods=10, freq="D")
    state = pd.DataFrame({"bar_index": np.arange(10),
                          "decision_time": times})
    split = _split_with_end("2020-01-05")
    m = R9B.test_window_mask(state, split)
    assert bool(m[2]) is True
    assert bool(m[4]) is True     # 2020-01-05 == COMMON_END -> in
    assert bool(m[5]) is False    # 2020-01-06 > COMMON_END -> out


# --------------------------------------------------------------------------- #
# G12 : missing R8 artifact hard-fails the formal runner                       #
# --------------------------------------------------------------------------- #
def test_formal_runner_missing_r8_artifact_hard_fails(monkeypatch, tmp_path):
    dec = tmp_path / "dec"
    dec.mkdir()
    ev = tmp_path / "pretest.json"
    # pretest references an R8 artifact that does NOT exist in the temp dir.
    pretest = {"r8_manifest": {"artifact_sha256": {"state_v1.parquet": "deadbeef"}}}
    ev.write_text(json.dumps(pretest))
    (dec / "r8_manifest_v1.json").write_text(
        json.dumps(pretest["r8_manifest"]))
    monkeypatch.setattr(R10, "DEC_ARTIFACT_DIR", str(dec))
    monkeypatch.setattr(R10, "PRETEST_SUMMARY_PATH", str(ev))
    head = R10._git_head_sha()
    with pytest.raises(RuntimeError) as exc:
        R10.run_formal_opportunity_value_test(
            allow_test=True, authorized_review_sha=head)
    assert "STOP_R10_R8_ARTIFACT_MISSING" in str(exc.value)


# --------------------------------------------------------------------------- #
# G17 : per-symbol point estimates use complete inference days only            #
# --------------------------------------------------------------------------- #
def test_per_symbol_deltas_use_inference_days():
    n = 12
    base = pd.Series(np.arange(float(n)))
    per = {
        R10.BASELINE_POLICY: {s: base.copy() for s in ["A", "B"]},
        R10.GATE_POLICY: {s: base * 1.1 for s in ["A", "B"]},
        R10.PRIMARY_POLICY: {s: base * 1.2 for s in ["A", "B"]},
    }
    rows = R10.per_symbol_deltas(per, common_days=n)
    n_inf = int(R10.complete_blocks(n)[1])
    for r in rows:
        s = r["symbol"]
        p0 = per[R10.BASELINE_POLICY][s].to_numpy(float)[:n_inf]
        p2 = per[R10.PRIMARY_POLICY][s].to_numpy(float)[:n_inf]
        assert abs(r["delta_full_point"] - float((p2 - p0).mean())) < 1e-9
        assert r["n_inference_days"] == n_inf


# --------------------------------------------------------------------------- #
# G21 : Formal path isolation wiring                                          #
# --------------------------------------------------------------------------- #
def test_formal_paths_isolation():
    canon = R10._formal_paths(None)
    assert canon["evidence_dir"] == R10.FORMAL_EVIDENCE_DIR
    assert canon["test_pred"] == R10.TEST_PRED_PARQUET
    assert canon["e9_root"] == R10.E9_ROOT_AXIS_PARQUET

    tmp = "/tmp/_r10_mock_root"
    redir = R10._formal_paths(tmp)
    assert redir["evidence_dir"] == os.path.join(tmp, "evidence")
    assert redir["test_pred"] == os.path.join(tmp, "decomposed_predictions_test_v1.parquet")
    assert redir["e9_root"] == os.path.join(tmp, "e9_root_axis_v1.parquet")


# --------------------------------------------------------------------------- #
# F1 / F4 : PRE-TEST evidence -> local manifest -> model bytes binding          #
# --------------------------------------------------------------------------- #
def _real_manifest(sub):
    with open(os.path.join(R10._models_dir(sub), "model_manifest.json")) as f:
        return json.load(f)


def _write_bundle(root, sub, manifest, contents):
    """Write a model tree; returns (dir, {name: sha256})."""
    d = os.path.join(str(root), "models", sub)
    os.makedirs(d, exist_ok=True)
    for name, data in contents.items():
        with open(os.path.join(d, name), "w") as f:
            f.write(data)
    with open(os.path.join(d, "model_manifest.json"), "w") as f:
        json.dump(manifest, f)
    return d, {n: R10.sha256_file(os.path.join(d, n)) for n in contents}


def _mock_pretest(tmp_path, pre_win, pre_pay):
    """Redirect the formal runner at a temp PRE-TEST + empty R8 artifact set."""
    dec = tmp_path / "dec"
    dec.mkdir()
    (dec / "r8_manifest_v1.json").write_text(json.dumps({"artifact_sha256": {}}))
    ev = tmp_path / "pretest.json"
    ev.write_text(json.dumps({
        "r8_manifest": {"artifact_sha256": {}},
        "r9a": {"model_manifest": pre_win},
        "r9b": {"model_manifest": pre_pay},
    }))
    return dec, ev


def test_f4_binding_rejects_self_consistent_local_bundle(monkeypatch, tmp_path):
    """F4: local manifest + local files agree with each other but differ from
    the frozen PRE-TEST bundle -> must hard-fail before any TEST work."""
    pre_win = _real_manifest("win")
    pre_pay = _real_manifest("payoff")
    dec, ev = _mock_pretest(tmp_path, pre_win, pre_pay)

    # Local side ("B"): every local file really hashes to its local manifest
    # entry, so the pair is self-consistent, but it is NOT the frozen bundle.
    names = sorted(pre_win["model_sha256"])
    loc = json.loads(json.dumps(pre_win))
    _d, local_sha = _write_bundle(
        tmp_path / "art", "win", loc, {n: f"LOCAL-BYTES-{n}" for n in names})
    loc["model_sha256"] = dict(local_sha)
    with open(os.path.join(_d, "model_manifest.json"), "w") as f:
        json.dump(loc, f)
    # Sanity: the local pair really is self-consistent.
    assert all(R10.sha256_file(os.path.join(_d, n)) == local_sha[n] for n in names)

    monkeypatch.setattr(R10, "DEC_ARTIFACT_DIR", str(dec))
    monkeypatch.setattr(R10, "PRETEST_SUMMARY_PATH", str(ev))
    with pytest.raises(RuntimeError) as exc:
        R10.run_formal_opportunity_value_test(
            allow_test=True, authorized_review_sha=R10._git_head_sha(),
            art_root=str(tmp_path / "art"))
    assert "STOP_R10_R9A_PRETEST_BINDING_MISMATCH" in str(exc.value)
    # Rejected BEFORE TEST state load / TEST prediction / simulation.
    assert R10.COUNTERS["test_state_loads"] == 0
    assert R10.COUNTERS["test_prediction_loads"] == 0


def test_f4_binding_rejects_wrong_model_bytes(monkeypatch, tmp_path):
    """F4: manifest matches PRE-TEST but the model bytes do not -> hard fail."""
    pre_win = _real_manifest("win")
    pre_pay = _real_manifest("payoff")
    dec, ev = _mock_pretest(tmp_path, pre_win, pre_pay)

    # Local manifest is IDENTICAL to PRE-TEST; only the bytes differ.
    _write_bundle(tmp_path / "art", "win", json.loads(json.dumps(pre_win)),
                  {n: f"WRONG-BYTES-{n}" for n in sorted(pre_win["model_sha256"])})

    monkeypatch.setattr(R10, "DEC_ARTIFACT_DIR", str(dec))
    monkeypatch.setattr(R10, "PRETEST_SUMMARY_PATH", str(ev))
    with pytest.raises(RuntimeError) as exc:
        R10.run_formal_opportunity_value_test(
            allow_test=True, authorized_review_sha=R10._git_head_sha(),
            art_root=str(tmp_path / "art"))
    assert "STOP_R10_R9A_PRETEST_BINDING_MISMATCH" in str(exc.value)
    assert "model_bytes" in str(exc.value)
    assert R10.COUNTERS["test_state_loads"] == 0
    assert R10.COUNTERS["test_prediction_loads"] == 0


def test_f4_binding_accepts_identical_bundle(tmp_path):
    """F4 positive control: a byte-identical bundle is accepted and audited."""
    import shutil
    art = tmp_path / "art"
    for sub in ("win", "payoff"):
        src = R10._models_dir(sub)
        dst = os.path.join(str(art), "models", sub)
        os.makedirs(dst, exist_ok=True)
        for fn in os.listdir(src):
            shutil.copy2(os.path.join(src, fn), os.path.join(dst, fn))
    with open(R10.PRETEST_SUMMARY_PATH) as f:
        pre = json.load(f)

    audit = R10.verify_r9_pretest_binding(pre, art_root=str(art))
    assert set(audit) == {"R9A", "R9B"}
    assert audit["R9A"]["model_count"] == 3
    assert audit["R9B"]["model_count"] == 6
    assert sorted(audit["R9A"]["models_verified"]) == [
        "td1_win.txt", "td3_win.txt", "td5_win.txt"]
    assert sorted(audit["R9B"]["models_verified"]) == [
        "td1_loss_mag.txt", "td1_win_mag.txt", "td3_loss_mag.txt",
        "td3_win_mag.txt", "td5_loss_mag.txt", "td5_win_mag.txt"]
    # The audit SHAs are exactly the PRE-TEST frozen ones.
    for tag, key in (("R9A", "r9a"), ("R9B", "r9b")):
        assert audit[tag]["models_verified"] == \
            pre[key]["model_manifest"]["model_sha256"]
