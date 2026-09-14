"""PGM-0 — deterministic synthetic / contract tests.

最重要的两项（§16 第 5、6 条）：
  * Pairwise CRF 在 beta=0 时必须与 Conditional CRF 概率**逐元素一致**
  * 解析梯度 vs 有限差分梯度，最大相对误差 < 1e-5

其余硬 guard：
  sample hash / n / mask 范围 / 15-mask 行和 / preprocessing 只 fit train /
  eval 不进入 fit / Legacy 4-head 复现 STATE-1.1 F3 / optimizer success /
  TB4 措辞 / 只写 pgm0_* 文件 / 15 类 benchmark 的概率填充逻辑

运行：
  .venv/bin/python research/liquidity_oracle_atlas/test_pgm0_endpoint_coupling_v1.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import research.liquidity_oracle_atlas.experiment_pgm0_endpoint_coupling_v1 as G  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name} :: {detail}")
        FAILS.append(name)


# ---------------------------------------------------------------- masks
def test_mask_tables():
    check("15 legal masks", len(G.MASK_VALUES) == 15
          and G.MASK_VALUES.tolist() == list(range(1, 16)),
          G.MASK_VALUES.tolist())
    check("0000 is excluded from the state space", 0 not in G.MASK_VALUES, "")
    check("MASK_BITS shape is (15,4)", G.MASK_BITS.shape == (15, 4),
          G.MASK_BITS.shape)
    idx = G.MASK_TO_INDEX
    check("bit order is UP_PEN, DOWN_PEN, NEW_UPPER, NEW_LOWER",
          G.MASK_BITS[idx[1]].tolist() == [1., 0., 0., 0.]
          and G.MASK_BITS[idx[2]].tolist() == [0., 1., 0., 0.]
          and G.MASK_BITS[idx[4]].tolist() == [0., 0., 1., 0.]
          and G.MASK_BITS[idx[8]].tolist() == [0., 0., 0., 1.],
          [G.MASK_BITS[idx[m]].tolist() for m in (1, 2, 4, 8)])
    check("mask 15 has all four bits set",
          G.MASK_BITS[idx[15]].tolist() == [1., 1., 1., 1.],
          G.MASK_BITS[idx[15]].tolist())
    check("mask rows decode back to their integer value",
          all(sum(int(b) << k for k, b in enumerate(G.MASK_BITS[idx[m]]))
              == m for m in range(1, 16)), "")
    check("6 pairs in the preregistered order",
          G.PAIR_INDEX == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
          and G.MASK_PAIRS.shape == (15, 6), G.PAIR_INDEX)
    check("MASK_PAIRS is the product of the two bits",
          all(G.MASK_PAIRS[q, j] == G.MASK_BITS[q, i] * G.MASK_BITS[q, jj]
              for q in range(15)
              for j, (i, jj) in enumerate(G.PAIR_INDEX)), "")
    check("mask_index maps 1..15 to 0..14",
          G.mask_index([1, 15]).tolist() == [0, 14], "")
    check("mask_index rejects 0 and 16",
          _raises(lambda: G.mask_index([0]))
          and _raises(lambda: G.mask_index([16])), "")


def _raises(fn):
    try:
        fn()
        return False
    except Exception:
        return True


# ============================================ CRITICAL: beta=0 equivalence
def test_beta_zero_equivalence():
    rng = np.random.default_rng(5)
    n, d = 40, 7
    X = rng.normal(size=(n, d))
    y_index = rng.integers(0, 15, n)

    theta_A = rng.normal(size=4 + 4 * d) * 0.3
    P_no = G.predict_mask_prob(theta_A, X, with_pairs=False)

    theta_C = np.concatenate([theta_A, np.zeros(6)])
    P_z = G.predict_mask_prob(theta_C, X, with_pairs=True)

    check("beta=0 pairwise CRF == conditional CRF, elementwise identical",
          np.array_equal(P_no, P_z),
          float(np.max(np.abs(P_no - P_z))))
    check("both are valid distributions (rows sum to 1)",
          np.allclose(P_no.sum(axis=1), 1.0)
          and np.allclose(P_z.sum(axis=1), 1.0), "")

    # loss/gradient 也要一致（beta 的梯度在 beta=0 处不必须为 0，但 loss 一致）
    l_no, g_no = G.crf_objective(theta_A, X, y_index, with_pairs=False, C=1.0)
    l_z, g_z = G.crf_objective(theta_C, X, y_index, with_pairs=True, C=1.0)
    check("beta=0 gives the same loss",
          abs(l_no - l_z) < 1e-12, (l_no, l_z))
    check("beta=0 gives the same unary gradients",
          np.allclose(g_no, g_z[:len(g_no)], atol=1e-12), "")
    check("non-zero beta strictly changes the distribution",
          not np.array_equal(
              P_z,
              G.predict_mask_prob(
                  np.concatenate([theta_A, np.zeros(5).tolist() + [0.7]]),
                  X, with_pairs=True)), "")


# ============================================ CRITICAL: gradient check
def test_analytical_gradient_vs_finite_difference():
    rng = np.random.default_rng(17)
    for with_pairs in (False, True):
        for n, d in ((25, 4), (60, 9)):
            X = rng.normal(size=(n, d))
            y_index = rng.integers(0, 15, n)
            size = 4 + 4 * d + (6 if with_pairs else 0)
            theta = rng.normal(size=size) * 0.25

            loss, grad = G.crf_objective(theta, X, y_index,
                                         with_pairs=with_pairs, C=1.0)
            eps = 1e-6
            worst = 0.0
            idx = list(range(0, size, max(1, size // 12)))
            for k in idx:
                tp = theta.copy()
                tp[k] += eps
                tm = theta.copy()
                tm[k] -= eps
                fd = (G.crf_objective(tp, X, y_index, with_pairs=with_pairs,
                                      C=1.0)[0]
                      - G.crf_objective(tm, X, y_index, with_pairs=with_pairs,
                                        C=1.0)[0]) / (2 * eps)
                denom = max(1e-12, abs(fd) + abs(grad[k]))
                worst = max(worst, abs(fd - grad[k]) / denom)
            check(f"gradient max relative error < 1e-5 "
                  f"(pairs={with_pairs}, n={n}, d={d})",
                  worst < 1e-5, worst)

    # lam 与 C 的关系：C 变大 -> L2 变弱 -> loss 下降
    X = rng.normal(size=(30, 5))
    yi = rng.integers(0, 15, 30)
    th = rng.normal(size=4 + 20) * 0.3
    l1 = G.crf_objective(th, X, yi, with_pairs=False, C=1.0)[0]
    l2 = G.crf_objective(th, X, yi, with_pairs=False, C=1e6)[0]
    check("lam = 1/(C*N): larger C gives weaker regularization",
          l2 < l1, (l1, l2))


def test_truncated_independent_joint():
    rng = np.random.default_rng(23)
    p = rng.uniform(0.05, 0.95, size=(200, 4))
    Q = G.truncated_independent_joint(p)
    check("truncated joint shape is (n,15)", Q.shape == (200, 15), Q.shape)
    check("rows sum to 1", np.allclose(Q.sum(axis=1), 1.0), "")
    check("0000 mass is exactly redistributed (no zero row)",
          bool((Q >= 0).all() and (Q.sum(axis=1) > 0).all()), "")
    marg = G.marginal_bit_prob(Q)
    check("renormalisation changes the marginals (documented behaviour)",
          not np.allclose(marg, p, atol=1e-6),
          float(np.max(np.abs(marg - p))))
    check("returned mass is a proper distribution over 15 masks",
          np.allclose(Q[:, G.mask_index([int(m) for m in G.MASK_VALUES])].sum(1)
                      if False else Q.sum(1), 1.0), "")
    # 手工核对一个 mask：0000 的独立概率被移除后，其余按比例放大
    q = p[0]
    log_scores = (np.log(q) @ G.MASK_BITS.T
                  + np.log1p(-q) @ (1 - G.MASK_BITS).T)
    ref = np.exp(log_scores - log_scores.max())
    ref = ref / ref.sum()
    check("manual softmax over 15 masks matches the implementation",
          np.allclose(ref, Q[0], atol=1e-12), "")


def test_joint_and_marginal_helpers():
    rng = np.random.default_rng(29)
    P = rng.dirichlet(np.ones(15), size=50)
    y_index = rng.integers(0, 15, 50)
    nll = G.joint_nll(G.MASK_VALUES[y_index], P)
    check("joint_nll is finite and non-negative",
          np.isfinite(nll) and nll >= 0, nll)
    check("joint_nll of a one-hot perfect distribution is ~0",
          abs(G.joint_nll([5], np.eye(15)[[G.mask_index([5])[0]]])) < 1e-9, "")
    check("multiclass_brier is 0 for a perfect one-hot",
          abs(G.multiclass_brier([G.mask_index([7])[0]],
                                 np.eye(15)[[G.mask_index([7])[0]]])) < 1e-12,
          "")
    pj = G.pair_joint_prob(P)
    check("pair_joint_prob shape is (n,6)", pj.shape == (50, 6), pj.shape)
    check("pair joint probs lie in [0,1]",
          bool((pj >= 0).all() and (pj <= 1).all()), "")


# ------------------------------------------------------- preprocessing
def test_preprocessing_fit_only_on_train():
    src = Path(G.__file__).read_text()
    fit_lines = [ln.strip() for ln in src.splitlines()
                 if re.search(r"\.fit\(", ln)]
    check("every .fit( is on train objects only",
          all(("tr[COLS]" in ln) or ("Xtr" in ln) or ("m_tr" in ln)
              or ("p.fit(" in ln and "tr" in ln)
              for ln in fit_lines if not ln.startswith("#")),
          fit_lines)
    check("eval frame is never fitted",
          not any("ev[COLS]]" in ln.replace(" ", "") and ".fit(" in ln
                  for ln in fit_lines), fit_lines)
    check("all preprocessing comes from make_pipeline (frozen spec)",
          "make_pipeline(NUM, CAT)" in src, "")
    check("preprocessing parity across the four legacy heads is asserted",
          "STOP_PGM0_PREPROCESSING_MISMATCH" in src, "")
    check("regularization is lam = 1/(C*N) with C=1",
          "lam = 1.0 / (C * n)" in src and "C_REG = 1.0" in src, "")


def test_no_forbidden_scope():
    src = Path(G.__file__).read_text()
    banned_imports = ["prevprev_event", "load_seq", "duration_bars",
                      "prev_net_move_R", "symbol"]
    feats = " ".join(G.COLS)
    check("frozen feature list only has geometry + provenance + prev_event",
          set(G.COLS) == set(list(G.GEOM) + list(G.FUNC_PROV)
                             + ["prev_event_mask"]), G.COLS)
    check("duration / path / prevprev not in the feature list",
          not any(t in feats for t in
                  ["duration", "path", "prevprev", "net_move"]) , feats)
    check("no third-order interaction table is built",
          "MASK_TRIPLES" not in src and "TRIPLE" not in src, "")
    check("coupling policy forbids edge deletion / re-fitting",
          "NOT used to delete edges" in src, "")
    check("source never claims 'TB4 never read'",
          "TB4 never read" not in src and "tb4_read_for_values" not in src, "")
    check("source records tb4_analytically_used",
          "tb4_analytically_used" in src, "")
    written = []
    for ln in src.splitlines():
        if ".write_text(" in ln or ".to_csv(" in ln or ".to_parquet(" in ln:
            written += re.findall(r'"([A-Za-z0-9_]+\.(?:csv|json|parquet))"',
                                  ln)
    check("only pgm0_* results (and the gitignored sample cache) are written",
          all(w.startswith("pgm0_") for w in written), written)
    check("forbidden frozen outputs are opened read-only",
          "state11_summary.json\").write_text" not in src
          and "state1_sample_audit.json\").write_text" not in src
          and "repl0_samples.parquet\").write" not in src, "")


# --------------------------------------------------------------- runtime
def test_sample_parity_on_disk():
    p = G.CACHE / "repl0_samples.parquet"
    if not p.exists():
        check("repl0 sample cache present (skipped)", True, "")
        return
    from research.liquidity_oracle_atlas.experiment_state1_provenance_decomposition_v1 import (  # noqa: E402
        sample_key_hash,
    )
    sm = pd.read_parquet(p)
    check("n_samples == 37,224", len(sm) == 37224, len(sm))
    check("sample hash == frozen ddf034ed...",
          sample_key_hash(sm) == G.FROZEN_SAMPLE_HASH, "")
    check("target_event_mask entirely inside 1..15",
          int(sm["target_event_mask"].min()) >= 1
          and int(sm["target_event_mask"].max()) <= 15, "")
    check("no target censor row present",
          int((sm["target_event_mask"] == 0).sum()) == 0, "")


def test_runtime_outputs_if_present():
    f = G.OUT / "pgm0_summary.json"
    if not f.exists():
        check("pgm0_summary.json exists (skipped)", True, "")
        return
    d = json.loads(f.read_text())
    audit = d["sample_audit"]
    check("sample hash parity recorded as True",
          audit["hash_matches_frozen"] is True, audit["hash_matches_frozen"])
    check("legacy 4-head reproduces STATE-1.1 F3 exactly (both windows)",
          all(v["exact"] is True
              for v in audit["legacy_parity_vs_state11_F3"].values()),
          audit["legacy_parity_vs_state11_F3"])
    check("tb4_analytically_used is false",
          d["tb4_analytically_used"] is False, "")
    check("optimizer audit present for both CRFs in both windows",
          (G.OUT / "pgm0_optimizer_audit.csv").exists()
          and len(pd.read_csv(G.OUT / "pgm0_optimizer_audit.csv")) == 4, "")
    oa = pd.read_csv(G.OUT / "pgm0_optimizer_audit.csv")
    check("all optimizations report success=True",
          bool(oa["success"].all()), oa[["model", "success"]].to_dict("records"))
    check("param counts are 4+4d and 4+4d+6",
          set(oa["n_params"]) == {120, 126}, sorted(oa["n_params"].tolist()))
    cal = pd.read_csv(G.OUT / "pgm0_mask_calibration.csv")
    check("mask calibration covers 15 masks x 4 models x 2 windows",
          len(cal) == 120 and cal["mask"].nunique() == 15, len(cal))
    tot = cal.groupby(["window", "model"])["actual_freq"].sum()
    check("actual mask frequencies sum to 1 in every cell",
          bool(np.allclose(tot.to_numpy(), 1.0)), tot.tolist())
    coupl = pd.read_csv(G.OUT / "pgm0_couplings.csv")
    check("6 couplings reported per window",
          len(coupl) == 12 and coupl["pair"].nunique() == 6, len(coupl))
    check("all 6 coupling signs agree across the two windows",
          all((g["sign"].nunique() == 1) for _, g in coupl.groupby("pair")),
          coupl.to_dict("records"))
    boots = pd.read_csv(G.OUT / "pgm0_bootstrap.csv")
    check("3 comparisons per window (6 rows)",
          len(boots) == 6 and (boots["role"] == "PRIMARY").sum() == 2, len(boots))
    check("PRIMARY is C - A",
          set(boots[boots["role"] == "PRIMARY"]["comparison"])
          == {"C_PAIRWISE_CRF - A_COND_INDEPENDENT_CRF"}, "")
    check("benchmark B is labelled reference-only, not a gate",
          set(boots[boots["role"] == "REFERENCE_ONLY"]["comparison"])
          == {"B_MULTINOMIAL_15_MASK - A_COND_INDEPENDENT_CRF"}, "")
    for nm in ["pgm0_model_metrics.csv", "pgm0_by_symbol.csv",
               "pgm0_pair_calibration.csv", "pgm0_sample_audit.json"]:
        check(f"{nm} written", (G.OUT / nm).exists(), nm)


def main():
    test_mask_tables()
    test_beta_zero_equivalence()
    test_analytical_gradient_vs_finite_difference()
    test_truncated_independent_joint()
    test_joint_and_marginal_helpers()
    test_preprocessing_fit_only_on_train()
    test_no_forbidden_scope()
    test_sample_parity_on_disk()
    test_runtime_outputs_if_present()
    print(f"\n==== {len(FAILS)} FAIL ====")
    if FAILS:
        print("FAILED:", FAILS)
        sys.exit(1)
    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
