"""PGM-0 — Joint endpoint dependence (pairwise conditional CRF)

===========================================================================
唯一研究问题
===========================================================================
控制 current geometry、functional boundary provenance 和 immediately previous
structural endpoint 之后，当前 episode 的四个 endpoint bits 是否仍存在稳定的
**同期 pairwise conditional dependence**？

    X = (G, P, E_{n-1})      固定 4 geometry + 10 functional provenance
                             + prev_event_mask
    Y = (UP_PEN, DOWN_PEN, NEW_UPPER, NEW_LOWER)

episode 一定以 structural event 结束，所以 Y != 0000，合法状态空间
Omega = {0,1}^4 \\ {0000}，共 15 个 mask。

四个模型（同一 train-only preprocessing）：

    L  LEGACY_4_HEADS          现有四个独立 Logistic（= STATE-1.1 F3）后做
                               截断独立 joint（排除 0000 重新归一化）
    A  COND_INDEPENDENT_CRF    beta = 0 的 joint CRF，严格嵌套基准
    B  MULTINOMIAL_15_MASK     15 类 softmax，只作上限参考，不作 gate
    C  PAIRWISE_CRF            A + 6 个全局常数 beta

    PRIMARY  = C - A   （严格 nested：A 是 beta=0）
    secondary= C - L   （实际模型价值）

目标函数（平均 NLL + 固定 L2，与 C=1 同量级）：
    loss = -mean( log P(Y_n | X_n) ) + 0.5*lam*(||W||^2 + ||beta||^2)
    lam  = 1 / (C * N)            intercept 不惩罚

Windows：A: fit TB1 -> eval TB2 (seed 20260922)
         B: fit TB1+TB2 -> eval TB3 (seed 20260923)
bootstrap 按 target start trading day，1000 reps。

样本：repl0_samples.parquet n=37,224，
hash 必须 == ddf034ed30a90597aa86dbad71d6397b1934d532afb348b8a93b406984787c12。

禁止：duration / path / prevprev_event / SMC / symbol / volume /
latent state / HMM / RL / PnL / feature search / 三阶 interaction。
禁止根据 beta 删边、挑边、重新训练。

TB4：tb4_analytically_used = false。

===========================================================================
输出
===========================================================================
    pgm0_summary.json
    pgm0_sample_audit.json
    pgm0_model_metrics.csv
    pgm0_bootstrap.csv
    pgm0_by_symbol.csv
    pgm0_mask_calibration.csv
    pgm0_pair_calibration.csv
    pgm0_couplings.csv
    pgm0_optimizer_audit.csv
（大型 pgm0_samples.parquet 存 gitignored cache）
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.liquidity_oracle_atlas.experiment_local_liquidity_transition_v0 import (  # noqa: E402
    FULL_UNIV, TEST_BLOCK, OUT, CACHE, build_blocks,
)
from research.liquidity_oracle_atlas.run_5m_graph_probability_v1 import (  # noqa: E402
    load_raw_bars,
)
from research.liquidity_oracle_atlas.experiment_path0_episode_path_memory_v1 import (  # noqa: E402
    BIT_NAMES, BIT_MASKS, ece_binary, make_pipeline,
)
from research.liquidity_oracle_atlas.experiment_state1_provenance_decomposition_v1 import (  # noqa: E402
    GEOM, group_provenance, sample_key_hash,
)
from research.liquidity_oracle_atlas.experiment_step01_cause_decomposition_v1 import (  # noqa: E402
    load_seq,
)
from research.liquidity_oracle_atlas.experiment_state11_functional_provenance_v1 import (  # noqa: E402
    FUNC_PROV, FROZEN_SAMPLE_HASH,
)

REPL_BLOCK = "TB3"
CAT = ["prev_event_mask"]
NUM = list(GEOM) + list(FUNC_PROV)
assert len(NUM) == 14
COLS = NUM + CAT
C_REG = 1.0

# ---------------------------------------------------------------- 15 masks
MASK_VALUES = np.arange(1, 16, dtype=np.int64)

MASK_BITS = np.array(
    [[(m >> k) & 1 for k in range(4)] for m in MASK_VALUES],
    dtype=np.float64,
)  # (15, 4)  bit0 UP_PEN, bit1 DOWN_PEN, bit2 NEW_UPPER, bit3 NEW_LOWER

PAIR_INDEX = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
PAIR_NAMES = [f"{BIT_NAMES[i]} x {BIT_NAMES[j]}" for i, j in PAIR_INDEX]

MASK_PAIRS = np.array(
    [[y[i] * y[j] for i, j in PAIR_INDEX] for y in MASK_BITS],
    dtype=np.float64,
)  # (15, 6)

MASK_TO_INDEX = {int(m): i for i, m in enumerate(MASK_VALUES)}

WINDOWS = [
    dict(name="A_TB1_to_TB2", train=["TB1"], eval=TEST_BLOCK, seed=20260922),
    dict(name="B_TB1TB2_to_TB3", train=["TB1", "TB2"], eval=REPL_BLOCK,
         seed=20260923),
]
BOOTSTRAP_REPS = 1000


def mask_index(mask):
    mask = np.asarray(mask, dtype=np.int64)
    if np.any((mask < 1) | (mask > 15)):
        raise ValueError("endpoint mask 必须在 1..15")
    return np.array([MASK_TO_INDEX[int(m)] for m in mask], dtype=np.int64)


# ============================================================ 参数布局
def unpack(theta, d, with_pairs):
    pos = 0
    intercept = theta[pos:pos + 4]
    pos += 4
    W = theta[pos:pos + 4 * d].reshape(4, d)
    pos += 4 * d
    if with_pairs:
        beta = theta[pos:pos + 6]
    else:
        beta = np.zeros(6, dtype=float)
    return intercept, W, beta


def crf_objective(theta, X, y_index, *, with_pairs, C=1.0):
    """返回 (loss, gradient)。X 为 train-only preprocessing 后的 dense 设计矩阵。"""
    n, d = X.shape
    intercept, W, beta = unpack(theta, d, with_pairs)

    eta = intercept[None, :] + X @ W.T          # (n, 4)
    scores = eta @ MASK_BITS.T                  # (n, 15)

    if with_pairs:
        scores = scores + (MASK_PAIRS @ beta)[None, :]

    log_z = logsumexp(scores, axis=1)
    obs_score = scores[np.arange(n), y_index]
    nll = np.mean(log_z - obs_score)

    lam = 1.0 / (C * n)
    reg = 0.5 * lam * np.sum(W * W)
    if with_pairs:
        reg += 0.5 * lam * np.sum(beta * beta)
    loss = nll + reg

    prob = np.exp(scores - log_z[:, None])

    obs_bits = MASK_BITS[y_index]
    exp_bits = prob @ MASK_BITS
    diff_bits = exp_bits - obs_bits

    grad_intercept = diff_bits.mean(axis=0)
    grad_W = diff_bits.T @ X / n
    grad_W += lam * W

    parts = [grad_intercept.ravel(), grad_W.ravel()]
    if with_pairs:
        obs_pairs = MASK_PAIRS[y_index]
        exp_pairs = prob @ MASK_PAIRS
        grad_beta = (exp_pairs - obs_pairs).mean(axis=0)
        grad_beta += lam * beta
        parts.append(grad_beta)

    return float(loss), np.concatenate(parts)


def fit_conditional_crf(X, mask, *, with_pairs):
    y_index = mask_index(mask)
    n, d = X.shape
    size = 4 + 4 * d + (6 if with_pairs else 0)
    theta0 = np.zeros(size, dtype=float)

    # scipy 1.18 的 minimize 不接受 kwargs=，用闭包（数学模型不变）
    def fun(th):
        return crf_objective(th, X, y_index, with_pairs=with_pairs, C=C_REG)

    t0 = time.perf_counter()
    result = minimize(fun, theta0, method="L-BFGS-B", jac=True,
                      options=dict(maxiter=2000, ftol=1e-12, gtol=1e-8))
    elapsed = time.perf_counter() - t0
    if not result.success:
        raise SystemExit(
            f"STOP_PGM0_OPTIMIZER_FAIL: with_pairs={with_pairs} "
            f"{result.message}")
    audit = dict(success=bool(result.success), n_iter=int(result.nit),
                 final_fun=float(result.fun), message=str(result.message),
                 n_params=int(size), elapsed_seconds=round(elapsed, 2))
    return result.x, audit


def predict_mask_prob(theta, X, *, with_pairs):
    n, d = X.shape
    intercept, W, beta = unpack(theta, d, with_pairs)
    eta = intercept[None, :] + X @ W.T
    scores = eta @ MASK_BITS.T
    if with_pairs:
        scores = scores + (MASK_PAIRS @ beta)[None, :]
    log_z = logsumexp(scores, axis=1)
    return np.exp(scores - log_z[:, None])


def joint_nll(mask, prob):
    yi = mask_index(mask)
    q = np.clip(prob[np.arange(len(yi)), yi], 1e-15, 1.0)
    return float(-np.log(q).mean())


def marginal_bit_prob(mask_prob):
    return mask_prob @ MASK_BITS


def pair_joint_prob(mask_prob):
    return mask_prob @ MASK_PAIRS


def truncated_independent_joint(bit_prob):
    """把 0000 排除后重新归一化，得到 (n,15)。"""
    p = np.clip(bit_prob, 1e-15, 1 - 1e-15)
    log_p = np.log(p)
    log_q = np.log1p(-p)
    scores = log_p @ MASK_BITS.T + log_q @ (1.0 - MASK_BITS).T
    scores -= logsumexp(scores, axis=1)[:, None]
    return np.exp(scores)


# ============================================================ metrics
def multiclass_brier(y_index, prob):
    oh = np.zeros_like(prob)
    oh[np.arange(len(y_index)), y_index] = 1.0
    return float(np.mean(np.sum((prob - oh) ** 2, axis=1)))


def marginal_bit_brier(y_bit, p_bit):
    return float(np.mean((p_bit - y_bit) ** 2))


def pair_actual(y_bits, i, j):
    return (y_bits[:, i] * y_bits[:, j])


def densify(X):
    return np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=float)


# ============================================================ one window
def run_window(win, sm):
    tr = sm[sm["target_start_block"].isin(win["train"])].reset_index(drop=True)
    ev = sm[sm["target_start_block"] == win["eval"]].reset_index(drop=True)
    if not len(tr) or not len(ev):
        raise SystemExit(f"STOP_PGM0_EMPTY_SPLIT: {win['name']}")
    if win["eval"] in set(tr["target_start_block"]):
        raise SystemExit(f"STOP_PGM0_EVAL_IN_FIT: {win['name']}")
    if int(tr["target_event_mask"].min()) < 1 \
            or int(tr["target_event_mask"].max()) > 15:
        raise SystemExit("STOP_PGM0_MASK_RANGE")
    if int(ev["target_event_mask"].min()) < 1 \
            or int(ev["target_event_mask"].max()) > 15:
        raise SystemExit("STOP_PGM0_MASK_RANGE")

    y_tr_bits = np.stack([((tr["target_event_mask"].to_numpy() & m) != 0)
                          .astype(np.int64) for m in BIT_MASKS], axis=1)
    y_ev_bits = np.stack([((ev["target_event_mask"].to_numpy() & m) != 0)
                          .astype(np.int64) for m in BIT_MASKS], axis=1)
    m_tr = tr["target_event_mask"].to_numpy(np.int64)
    m_ev = ev["target_event_mask"].to_numpy(np.int64)
    yi_ev = mask_index(m_ev)

    # ---------- L: legacy 4 heads（与 STATE-1.1 F3 逐位一致） ----------
    t0 = time.perf_counter()
    pipes = []
    for k in range(4):
        p = make_pipeline(NUM, CAT)
        p.fit(tr[COLS], y_tr_bits[:, k])
        pipes.append(p)
    legacy_seconds = time.perf_counter() - t0

    pres = [p.named_steps["pre"] for p in pipes]
    Xtr = densify(pres[0].transform(tr[COLS]))
    Xev = densify(pres[0].transform(ev[COLS]))
    for p in pres[1:]:
        if not np.allclose(densify(p.transform(tr[COLS])), Xtr):
            raise SystemExit("STOP_PGM0_PREPROCESSING_MISMATCH")
        if not np.allclose(densify(p.transform(ev[COLS])), Xev):
            raise SystemExit("STOP_PGM0_PREPROCESSING_MISMATCH")

    bit_ev = np.stack([p.predict_proba(ev[COLS])[:, 1] for p in pipes], axis=1)
    P_legacy = truncated_independent_joint(bit_ev)

    # ----------------------- A / C: conditional CRF --------------------
    t0 = time.perf_counter()
    theta_A, aud_A = fit_conditional_crf(Xtr, m_tr, with_pairs=False)
    opt_seconds_A = time.perf_counter() - t0
    t0 = time.perf_counter()
    theta_C, aud_C = fit_conditional_crf(Xtr, m_tr, with_pairs=True)
    opt_seconds_C = time.perf_counter() - t0
    P_A = predict_mask_prob(theta_A, Xev, with_pairs=False)
    P_C = predict_mask_prob(theta_C, Xev, with_pairs=True)
    _, _, beta_C = unpack(theta_C, Xev.shape[1], True)

    # ---------------------- B: 15-class benchmark ----------------------
    t0 = time.perf_counter()
    clf = LogisticRegression(penalty="l2", C=C_REG, solver="lbfgs",
                             max_iter=3000, class_weight=None)
    clf.fit(Xtr, m_tr)
    raw = clf.predict_proba(Xev)
    P_B = np.full((len(Xev), 15), 1e-15)
    for j, cls in enumerate(clf.classes_):
        P_B[:, int(cls) - 1] = raw[:, j]
    P_B /= P_B.sum(axis=1, keepdims=True)
    b_seconds = time.perf_counter() - t0

    PROBS = {"L_LEGACY_4_HEADS": P_legacy, "A_COND_INDEPENDENT_CRF": P_A,
             "B_MULTINOMIAL_15_MASK": P_B, "C_PAIRWISE_CRF": P_C}

    # ------------------------------ metrics ----------------------------
    #   legacy_head_bit_logloss：4 个 Logistic **原始**概率的 mean-bit logloss，
    #   这是与 STATE-1.1 F3 逐位对应的量（截断 joint 会改变边际，故单列）。
    legacy_head_bll = float(np.mean(
        [legacy_bit_head_logloss(y_ev_bits[:, k], bit_ev[:, k])
         for k in range(4)]))
    rows = []
    for name, P in PROBS.items():
        marg = marginal_bit_prob(P)
        pj = pair_joint_prob(P)
        rows.append(dict(
            window=win["name"], model=name, n=len(ev),
            joint_nll=joint_nll(m_ev, P),
            multiclass_brier=multiclass_brier(yi_ev, P),
            head_bit_logloss=legacy_head_bll
            if name == "L_LEGACY_4_HEADS" else np.nan,
            marginal_bit_brier=float(np.mean(
                [marginal_bit_brier(y_ev_bits[:, k], marg[:, k])
                 for k in range(4)])),
            marginal_bit_logloss=float(np.mean(
                [ece_like_logloss(y_ev_bits[:, k], marg[:, k])
                 for k in range(4)])),
            marginal_bit_ece=float(np.mean(
                [ece_binary(y_ev_bits[:, k], marg[:, k]) for k in range(4)])),
            pair_joint_brier=float(np.mean(
                [marginal_bit_brier(pair_actual(y_ev_bits, i, j), pj[:, q])
                 for q, (i, j) in enumerate(PAIR_INDEX)])),
        ))
    mtab = pd.DataFrame(rows)

    # mask / pair calibration
    cal_rows, pair_cal_rows = [], []
    for name, P in PROBS.items():
        for q, mv in enumerate(MASK_VALUES):
            act = float((m_ev == int(mv)).mean())
            pred = float(P[:, q].mean())
            cal_rows.append(dict(window=win["name"], model=name,
                                 mask=int(mv), n_actual=int((m_ev == mv).sum()),
                                 actual_freq=act, mean_pred_prob=pred,
                                 abs_diff=abs(pred - act)))
        pj = pair_joint_prob(P)
        for q, (i, j) in enumerate(PAIR_INDEX):
            act = float(pair_actual(y_ev_bits, i, j).mean())
            pred = float(pj[:, q].mean())
            pair_cal_rows.append(dict(
                window=win["name"], model=name, pair=PAIR_NAMES[q],
                actual_joint_freq=act, mean_pred_joint_prob=pred,
                abs_diff=abs(pred - act)))

    # --------------------------- bootstrap -----------------------------
    t0 = time.perf_counter()
    day = ev["_eval_day"].to_numpy()
    uniq = np.unique(day)
    pos = np.searchsorted(uniq, day)
    nd = len(uniq)
    cnt = np.bincount(pos, minlength=nd)
    keep = cnt > 0
    nk = int(keep.sum())

    def per_sample_nll(mask, P):
        yi = mask_index(mask)
        return -np.log(np.clip(P[np.arange(len(yi)), yi], 1e-15, 1.0))

    nll = {n: per_sample_nll(m_ev, P) for n, P in PROBS.items()}

    def boot(dv, seed):
        r = np.random.default_rng(seed)
        b = np.empty(BOOTSTRAP_REPS)
        for i in range(BOOTSTRAP_REPS):
            b[i] = dv[r.integers(0, nk, nk)].mean()
        return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

    comps = [("C_PAIRWISE_CRF - A_COND_INDEPENDENT_CRF", "PRIMARY",
              "C_PAIRWISE_CRF", "A_COND_INDEPENDENT_CRF"),
             ("C_PAIRWISE_CRF - L_LEGACY_4_HEADS", "SECONDARY_PRACTICAL",
              "C_PAIRWISE_CRF", "L_LEGACY_4_HEADS"),
             ("B_MULTINOMIAL_15_MASK - A_COND_INDEPENDENT_CRF",
              "REFERENCE_ONLY", "B_MULTINOMIAL_15_MASK",
              "A_COND_INDEPENDENT_CRF")]
    boots = []
    for idx, (label, role, hi, lo) in enumerate(comps):
        dd = nll[hi] - nll[lo]
        dv = np.bincount(pos, weights=dd, minlength=nd)[keep] / cnt[keep]
        a, b = boot(dv, win["seed"] + idx)
        boots.append(dict(
            window=win["name"], comparison=label, role=role,
            joint_nll_A=float(mtab.loc[mtab["model"] == hi, "joint_nll"].iloc[0]),
            joint_nll_B=float(mtab.loc[mtab["model"] == lo, "joint_nll"].iloc[0]),
            delta_sample_weighted=float(dd.mean()),
            delta_daily_mean=float(dv.mean()), n_days=nk,
            ci_lo=a, ci_hi=b,
            verdict=("CI_below_zero" if b < 0 else
                     "CI_above_zero" if a > 0 else "CI_contains_zero")))
    boot_seconds = time.perf_counter() - t0

    by_sym = []
    dC_A = nll["C_PAIRWISE_CRF"] - nll["A_COND_INDEPENDENT_CRF"]
    dC_L = nll["C_PAIRWISE_CRF"] - nll["L_LEGACY_4_HEADS"]
    for s, g in ev.groupby("symbol"):
        idx = g.index.to_numpy()
        by_sym.append(dict(
            window=win["name"], symbol=s, n=int(len(g)),
            joint_nll_C=float(nll["C_PAIRWISE_CRF"][idx].mean()),
            joint_nll_A=float(nll["A_COND_INDEPENDENT_CRF"][idx].mean()),
            joint_nll_L=float(nll["L_LEGACY_4_HEADS"][idx].mean()),
            delta_C_minus_A=float(dC_A[idx].mean()),
            delta_C_minus_L=float(dC_L[idx].mean())))

    couplings = [dict(window=win["name"], pair=PAIR_NAMES[q],
                      beta=float(beta_C[q]), sign=int(np.sign(beta_C[q])))
                 for q in range(6)]

    optim = [dict(window=win["name"], model="A_COND_INDEPENDENT_CRF",
                  with_pairs=False, **aud_A,
                  wall_seconds=round(opt_seconds_A, 2)),
             dict(window=win["name"], model="C_PAIRWISE_CRF",
                  with_pairs=True, **aud_C,
                  wall_seconds=round(opt_seconds_C, 2))]

    return dict(win=win, n_train=int(len(tr)), n_eval=int(len(ev)),
                n_days=nk, mtab=mtab, boots=boots, by_sym=by_sym,
                couplings=couplings, optim=optim, cal_rows=cal_rows,
                pair_cal_rows=pair_cal_rows, dC_A=dC_A, dc_L=dC_L,
                legacy_bit=bit_ev, y_ev_bits=y_ev_bits, m_ev=m_ev,
                timing=dict(fit_legacy_seconds=round(legacy_seconds, 2),
                            fit_crf_A_seconds=round(opt_seconds_A, 2),
                            fit_crf_C_seconds=round(opt_seconds_C, 2),
                            fit_15mask_seconds=round(b_seconds, 2),
                            bootstrap_seconds=round(boot_seconds, 2)))


def ece_like_logloss(y, p):
    q = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log1p(-q)))


def legacy_bit_head_logloss(y, p):
    return ece_like_logloss(y, p)


def main():
    t_total = time.perf_counter()
    timing = {}

    # ------------------------------------------------- sample / hash parity
    t0 = time.perf_counter()
    sm = pd.read_parquet(CACHE / "repl0_samples.parquet")
    h = sample_key_hash(sm)
    if h != FROZEN_SAMPLE_HASH:
        raise SystemExit(f"STOP_PGM0_SAMPLE_PARITY_FAIL: {h}")
    if len(sm) != 37224:
        raise SystemExit(f"STOP_PGM0_SAMPLE_COUNT_FAIL: {len(sm)}")
    st11 = json.loads((OUT / "state11_summary.json").read_text())
    if st11["sample_parity"]["sample_key_sha256"] != FROZEN_SAMPLE_HASH:
        raise SystemExit("STOP_PGM0_FROZEN_CROSSCHECK_FAIL: STATE-1.1")

    # trading-day tag（仅用于 bootstrap 分组；block grid 是 LOCAL-0 冻结定义）
    bars = {s: load_raw_bars(s) for s in FULL_UNIV}
    all_days, day_block_code, boundaries = build_blocks(bars)
    td_by_sym, tb3_end_by_sym = {}, {}
    for s in FULL_UNIV:
        td_full = np.asarray(bars[s]["td"]).astype("datetime64[D]")
        code = day_block_code[np.searchsorted(all_days, td_full)]
        tb3_end = int(np.flatnonzero(code <= 2)[-1])
        td_by_sym[s] = td_full[:tb3_end + 1]
        tb3_end_by_sym[s] = tb3_end
    del bars
    sm = sm.copy()

    # ---- provenance（repl0_samples 不缓存 provenance）：用与 STATE-1 /
    #      1.1 / SEQ-1 完全相同的函数对象 group_provenance 确定性重算，
    #      再按 STATE-1.1 的公式生成 10 个 functional provenance 变量。
    #      重算结果与 STATE-1 已提交的 audit 逐项对齐；任何偏差都会被下面
    #      的 LEGACY-vs-STATE-1.1 F3 逐位 parity guard 再拦一次。
    st1 = json.loads((OUT / "state1_sample_audit.json").read_text())
    ep = pd.read_parquet(CACHE / "episode_repl0_through_tb3.parquet")
    kk = ep["symbol"].astype(str) + "|" + ep["start_bar"].astype(str)
    g_up = dict(zip(kk, ep["start_upper_group"].astype(np.int64)))
    g_dn = dict(zip(kk, ep["start_lower_group"].astype(np.int64)))
    grps = {s: load_seq(s)[1] for s in FULL_UNIV}
    prov = np.full((len(sm), 6), np.nan)
    for i, r in enumerate(sm.itertuples()):
        k = f"{r.symbol}|{r.target_start_bar}"
        pu = group_provenance(grps[r.symbol], int(g_up.get(k, -1)),
                              int(r.target_start_bar))
        pl = group_provenance(grps[r.symbol], int(g_dn.get(k, -1)),
                              int(r.target_start_bar))
        if pu is None or pl is None:
            raise SystemExit(
                "STOP_PGM0_PROVENANCE_CONTRACT_FAIL: "
                f"{r.symbol} t={r.target_start_bar}")
        prov[i, :] = [pu[0], pu[1], pu[2], pl[0], pl[1], pl[2]]
    RAW_PROV = ["upper_oldest_age_bars", "upper_newest_age_bars",
                "upper_n_active_identities", "lower_oldest_age_bars",
                "lower_newest_age_bars", "lower_n_active_identities"]
    for j, c in enumerate(RAW_PROV):
        sm[c] = prov[:, j]
    pa = st1["provenance"]
    chk = [
        (float(sm["upper_oldest_age_bars"].mean()),
         pa["upper_oldest_age_bars"]["mean"]),
        (float(sm["upper_oldest_age_bars"].median()),
         pa["upper_oldest_age_bars"]["p50"]),
        (float(sm["upper_oldest_age_bars"].max()),
         pa["upper_oldest_age_bars"]["max"]),
        (float(sm["upper_newest_age_bars"].mean()),
         pa["upper_newest_age_bars"]["mean"]),
        (float(sm["upper_newest_age_bars"].min()),
         pa["upper_newest_age_bars"]["min"]),
        (float(sm["upper_n_active_identities"].mean()),
         pa["upper_n_active_identities"]["mean"]),
        (float(sm["lower_oldest_age_bars"].mean()),
         pa["lower_oldest_age_bars"]["mean"]),
        (float(sm["lower_oldest_age_bars"].max()),
         pa["lower_oldest_age_bars"]["max"]),
        (float((sm["upper_newest_age_bars"] == 0).mean()),
         pa["frac_upper_newest_age_zero"]),
        (float((sm["lower_newest_age_bars"] == 0).mean()),
         pa["frac_lower_newest_age_zero"]),
    ]
    for got, exp in chk:
        if abs(got - exp) > 1e-9 * max(1.0, abs(exp)):
            raise SystemExit(
                "STOP_PGM0_PROVENANCE_FROZEN_MISMATCH: "
                f"{got} vs {exp}")
    for side in ("upper", "lower"):
        sm[f"{side}_oldest_log_age"] = np.log1p(
            sm[f"{side}_oldest_age_bars"].to_numpy(float))
        sm[f"{side}_newest_log_age"] = np.log1p(
            sm[f"{side}_newest_age_bars"].to_numpy(float))
        sm[f"{side}_newest_age_zero"] = (
            sm[f"{side}_newest_age_bars"].to_numpy() == 0).astype(float)
        sm[f"{side}_oldest_age_zero"] = (
            sm[f"{side}_oldest_age_bars"].to_numpy() == 0).astype(float)

    sm["_eval_day"] = np.array(
        [td_by_sym[s][b] for s, b in
         zip(sm["symbol"], sm["target_start_bar"])], dtype="datetime64[D]")
    for s, g in sm.groupby("symbol"):
        if int(g["target_start_bar"].max()) > tb3_end_by_sym[s]:
            raise SystemExit(f"STOP_PGM0_TB4_ENDPOINT: {s}")
    timing["cache_load_seconds"] = round(time.perf_counter() - t0, 2)

    # ------------------------------------------------------------ windows
    res = [run_window(w, sm) for w in WINDOWS]
    for k in ["fit_legacy_seconds", "fit_crf_A_seconds", "fit_crf_C_seconds",
              "fit_15mask_seconds", "bootstrap_seconds"]:
        timing[k] = round(sum(r["timing"][k] for r in res), 2)

    metrics = pd.concat([r["mtab"] for r in res], ignore_index=True)
    boots = pd.DataFrame([b for r in res for b in r["boots"]])
    by_sym = pd.DataFrame([b for r in res for b in r["by_sym"]])
    coupl = pd.DataFrame([c for r in res for c in r["couplings"]])
    optim = pd.DataFrame([o for r in res for o in r["optim"]])
    maskcal = pd.DataFrame([c for r in res for c in r["cal_rows"]])
    paircal = pd.DataFrame([c for r in res for c in r["pair_cal_rows"]])
    metrics.to_csv(OUT / "pgm0_model_metrics.csv", index=False)
    boots.to_csv(OUT / "pgm0_bootstrap.csv", index=False)
    by_sym.to_csv(OUT / "pgm0_by_symbol.csv", index=False)
    coupl.to_csv(OUT / "pgm0_couplings.csv", index=False)
    optim.to_csv(OUT / "pgm0_optimizer_audit.csv", index=False)
    maskcal.to_csv(OUT / "pgm0_mask_calibration.csv", index=False)
    paircal.to_csv(OUT / "pgm0_pair_calibration.csv", index=False)

    # --------------------------------------- legacy parity vs STATE-1.1
    f3 = {r["window"]: r for r in st11["results"]}
    legacy_parity = {}
    for r in res:
        w = r["win"]["name"]
        got = float(r["mtab"].loc[
            r["mtab"]["model"] == "L_LEGACY_4_HEADS",
            "head_bit_logloss"].iloc[0])
        exp = float(f3[w]["mean_bit_metrics"][
            "F3_GEOMETRY_FUNCTIONAL_PROVENANCE_PREV_EVENT"]["logloss"])
        legacy_parity[w] = dict(legacy_marginal_logloss=got,
                                state11_F3_logloss=exp,
                                exact=bool(abs(got - exp) < 1e-12))
        if not legacy_parity[w]["exact"]:
            raise SystemExit(
                f"STOP_PGM0_LEGACY_PARITY_FAIL: {w} {got} vs {exp}")

    # ------------------------------------------------------------ verdict
    prim = boots[boots["role"] == "PRIMARY"].set_index("window")
    both_ok = bool((prim["ci_hi"] < 0).all())
    none_inc = bool((prim["ci_lo"] >= 0).all())
    if both_ok:
        v = "PGM0_PAIRWISE_DEPENDENCE_SUPPORTED"
    elif none_inc:
        v = "PGM0_PAIRWISE_DEPENDENCE_NOT_STABLE"
    else:
        v = "PGM0_PAIRWISE_DEPENDENCE_NOT_STABLE"
    prac = boots[boots["role"] == "SECONDARY_PRACTICAL"].set_index("window")
    prac_ok = bool((prac["ci_hi"] < 0).all())

    audit = dict(
        n_samples=int(len(sm)),
        sample_key_sha256=h,
        hash_matches_frozen=bool(h == FROZEN_SAMPLE_HASH),
        target_mask_min=int(sm["target_event_mask"].min()),
        target_mask_max=int(sm["target_event_mask"].max()),
        n_mask_zero=int((sm["target_event_mask"] == 0).sum()),
        mask_frequency={str(int(k)): int(v) for k, v in
                        sm["target_event_mask"].value_counts()
                        .sort_index().items()},
        window_counts={r["win"]["name"]: dict(
            n_train=r["n_train"], n_eval=r["n_eval"], n_days=r["n_days"])
            for r in res},
        legacy_parity_vs_state11_F3=legacy_parity,
        tb4_analytically_used=False,
        tb4_endpoint_check="all target_start_bar <= per-symbol TB3 last bar",
    )
    (OUT / "pgm0_sample_audit.json").write_text(
        json.dumps(audit, indent=2, default=str))
    sm.to_parquet(CACHE / "pgm0_samples.parquet", index=False)

    summary = dict(
        experiment="PGM-0 joint endpoint dependence (pairwise CRF)",
        base="825e3f518a109a91dbc319ca5202d868cf0cad65",
        question=("after controlling current geometry, functional boundary "
                  "provenance and the immediately previous structural "
                  "endpoint, do the four endpoint bits still show stable "
                  "same-episode pairwise conditional dependence?"),
        state=dict(features=COLS, n_numeric=len(NUM), n_categorical=len(CAT),
                   legal_state_space="15 masks (0000 impossible by episode "
                                     "definition)",
                   regularization=f"lam = 1/(C*N), C={C_REG}, intercept "
                                  "unpenalized, no tuning"),
        models=dict(
            L="legacy 4 independent logistic heads (= STATE-1.1 F3) + "
              "truncated independent joint",
            A="conditional CRF with beta=0 (strict nested baseline)",
            B="15-class multinomial softmax (upper-bound reference only, "
              "NOT a gate)",
            C="A + 6 global-constant pairwise couplings beta"),
        windows=[dict(name=w["name"], train=w["train"], eval=w["eval"],
                      seed=w["seed"]) for w in WINDOWS],
        bootstrap_reps=BOOTSTRAP_REPS,
        sample_audit=audit,
        results=[dict(window=r["win"]["name"], n_train=r["n_train"],
                      n_eval=r["n_eval"], n_days=r["n_days"],
                      metrics=r["mtab"].to_dict(orient="records")) for r in res],
        bootstrap=boots.to_dict(orient="records"),
        couplings=coupl.to_dict(orient="records"),
        by_symbol=dict(
            n_rows=len(by_sym), n_symbol_per_window=15,
            window_A_n_negative=int(((by_sym["window"] == "A_TB1_to_TB2")
                                     & (by_sym["delta_C_minus_A"] < 0)).sum()),
            window_A_n_positive=int(((by_sym["window"] == "A_TB1_to_TB2")
                                     & (by_sym["delta_C_minus_A"] > 0)).sum()),
            window_B_n_negative=int(((by_sym["window"] == "B_TB1TB2_to_TB3")
                                     & (by_sym["delta_C_minus_A"] < 0)).sum()),
            window_B_n_positive=int(((by_sym["window"] == "B_TB1TB2_to_TB3")
                                     & (by_sym["delta_C_minus_A"] > 0)).sum())),
        coupling_note=("6 global-constant betas reported for sign/magnitude/"
                       "cross-window direction only; NOT used to delete edges, "
                       "pick edges or re-fit"),
        mask_calibration_summary={
            r["win"]["name"]: float(
                pd.DataFrame(r["cal_rows"]).query("model == 'C_PAIRWISE_CRF'")
                ["abs_diff"].mean()) for r in res},
        pair_calibration_summary={
            r["win"]["name"]: float(
                pd.DataFrame(r["pair_cal_rows"])
                .query("model == 'C_PAIRWISE_CRF'")["abs_diff"].mean())
            for r in res},
        practical_gain_supported=prac_ok,
        tb4_analytically_used=False,
        timing=timing,
        interpretation_limits=(
            "PGM-0 only tests same-episode pairwise conditional dependence of "
            "the four endpoint bits. It does NOT authorise third-order "
            "interactions, latent state, HMM/HSMM, SMC, next-state models, "
            "tradability or RL."),
    )
    summary["PGM0_VERDICT"] = v
    if both_ok:
        summary["PGM0_CONCLUSION"] = (
            "stable same-episode pairwise endpoint dependence exists beyond "
            "the current state; endpoint coupling is worth modelling.")
    elif none_inc:
        summary["PGM0_CONCLUSION"] = (
            "no same-episode pairwise dependence beyond the current state: "
            "keep four independent logistic heads; stop endpoint coupling "
            "complexity. Do NOT try third-order interactions.")
    else:
        summary["PGM0_CONCLUSION"] = (
            "pairwise dependence is not stable across windows: stop endpoint "
            "coupling complexity. Do NOT try third-order interactions.")
    summary["PGM0_PRACTICAL"] = (
        "PGM0_PRACTICAL_JOINT_GAIN_SUPPORTED" if prac_ok
        else "PGM0_PRACTICAL_JOINT_GAIN_NOT_SUPPORTED")
    timing["total_seconds"] = round(time.perf_counter() - t_total, 2)
    summary["timing"] = timing
    (OUT / "pgm0_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    print(f"[SAMPLE] {len(sm)} hash_match={h == FROZEN_SAMPLE_HASH} "
          f"mask_range=[{audit['target_mask_min']},{audit['target_mask_max']}]")
    print(f"[LEGACY PARITY] {legacy_parity}")
    print(f"[METRICS]\n{metrics.to_string(index=False)}")
    print(f"[BOOTSTRAP]\n{boots.to_string(index=False)}")
    print(f"[COUPLINGS]\n{coupl.to_string(index=False)}")
    print(f"[OPTIMIZER]\n{optim.to_string(index=False)}")
    print(f"[BY-SYMBOL] {summary['by_symbol']}")
    print(f"[TIMING] {timing}")
    print(f"[VERDICT] {v} / {summary['PGM0_PRACTICAL']}")
    print(f"[DONE] -> {OUT}")


if __name__ == "__main__":
    main()
