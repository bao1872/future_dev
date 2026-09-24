"""FUTURE-R9C-M15-MATHEMATICAL-COMPOSER-V1.

R9C: the Mathematical Composer. Contains NO learned model. It only combines the
independent R9A (Win Probability) and R9B (Payoff Ratio) predictions by
deterministic arithmetic:

    RR(Z)   = mu_W(Z) / mu_L(Z)
    p*(Z)   = mu_L / (mu_W + mu_L) = 1 / (1 + RR)
    EV_C(Z) = p * mu_W - (1 - p) * mu_L

and the two clean single-model ablations:

    EV_W(Z) = p_hat * mu_W0 - (1 - p_hat) * mu_L0      # vary Win, fix Payoff
    EV_R(Z) = p0 * mu_W_hat - (1 - p0) * mu_L_hat      # vary Payoff, fix Win

where the TRAIN priors (p0, mu_W0, mu_L0, RR0) are computed from TRAIN rows
ONLY and never from VAL or TEST (§20). Hard identity: EV_C > 0  <=>  p > p*.

The Composer also produces VAL descriptive diagnostics (5x5 Pwin x RR grid and
combined-EV deciles). These are diagnostic only and never become policy rules.
"""

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.build_decomposed_value_dataset_v1 import (
    HORIZONS,
    WIN33_COLS,
    PAY8_COLS,
    load_labels,
)
from research.liquidity_oracle_atlas import win_probability_model_v1 as win_mod
from research.liquidity_oracle_atlas import payoff_ratio_model_v1 as pay_mod

TASK_ID = "FUTURE-R9C-M15-MATHEMATICAL-COMPOSER-V1"

COUNTERS = {"model_fits": 0}


# --------------------------------------------------------------------------- #
# 1. Core composition                                                          #
# --------------------------------------------------------------------------- #
def compose_probability_and_payoff(p_win, mu_win, mu_loss, check_identity=True):
    """Combine p, mu_W, mu_L into RR / break-even / combined EV.

    Raises RuntimeError if the algebra EV_C>0 <=> p>p* is violated (finite denom).
    """
    p = np.asarray(p_win, dtype=float)
    mw = np.maximum(np.asarray(mu_win, dtype=float), 0.0)
    ml = np.maximum(np.asarray(mu_loss, dtype=float), 0.0)

    denom = mw + ml
    rr = np.where(ml > 0, mw / ml, np.inf)
    p_break_even = np.where(denom > 0, ml / denom, 1.0)
    ev_combined = p * mw - (1.0 - p) * ml

    if check_identity:
        finite = (denom > 0) & np.isfinite(rr)
        if finite.any():
            expect = p[finite] > p_break_even[finite]
            got = ev_combined[finite] > 0.0
            if not np.array_equal(expect, got):
                raise RuntimeError("STOP_R9C_EV_IDENTITY p>p* mismatches EV_C>0")

    return {
        "p_win": p,
        "mu_win": mw,
        "mu_loss": ml,
        "predicted_rr": rr,
        "p_break_even": p_break_even,
        "ev_combined": ev_combined,
        "take_combined": np.isfinite(ev_combined) & (ev_combined > 0.0),
    }


def compose_scores(p_hat, mu_w_hat, mu_l_hat, *, p_train_prior,
                   mu_w_train_prior, mu_l_train_prior):
    """Return EV_W (Win-only ablation), EV_R (Payoff-only ablation), EV_C."""
    p = np.asarray(p_hat, dtype=float)
    mw = np.asarray(mu_w_hat, dtype=float)
    ml = np.asarray(mu_l_hat, dtype=float)

    ev_w = p * mu_w_train_prior - (1.0 - p) * mu_l_train_prior
    ev_r = p_train_prior * mw - (1.0 - p_train_prior) * ml
    ev_c = p * mw - (1.0 - p) * ml
    return ev_w, ev_r, ev_c


# --------------------------------------------------------------------------- #
# 2. TRAIN priors (TRAIN only)                                                 #
# --------------------------------------------------------------------------- #
def compute_train_priors(stage: str = "train") -> dict:
    """Weighted TRAIN priors per horizon: p0, muW0, muL0, RR0."""
    lab = load_labels(stage)
    lab = lab[lab["bracket_eligible"] & (lab["sample_weight"] > 0)]
    out = {}
    for H in HORIZONS:
        q = lab[lab["horizon"] == H]
        y = q["episode_return_atr"].to_numpy(float)
        w = q["sample_weight"].to_numpy(float)
        win = y > 0
        p0 = float(np.average(win.astype(float), weights=w))
        mu_w0 = (float(np.average(y[win], weights=w[win]))
                 if win.any() else np.nan)
        mu_l0 = (float(np.average((-y)[~win], weights=w[~win]))
                 if (~win).any() else np.nan)
        rr0 = (float(mu_w0 / mu_l0) if (mu_l0 and mu_l0 > 0) else np.nan)
        out[H] = {"p0": p0, "muW0": mu_w0, "muL0": mu_l0, "RR0": rr0}
    return out


# --------------------------------------------------------------------------- #
# 3. VAL descriptive diagnostics (no thresholds)                               #
# --------------------------------------------------------------------------- #
def _quintile_idx(arr, k=5):
    arr = np.asarray(arr, dtype=float)
    return pd.qcut(pd.Series(arr).rank(method="first"), k,
                   labels=False).to_numpy()


def _safe_quintile(arr, k=5):
    arr = np.asarray(arr, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(len(arr), dtype=int)
    fill = np.nanmax(arr[finite]) if k > 1 else 0.0
    a = np.where(finite, arr, fill)
    return _quintile_idx(a, k)


def _cell_stats(mask, p_hat, rr, y, w, pr):
    if not mask.any():
        return None
    ww = w[mask]
    yy = y[mask]
    win = yy > 0
    avg_win = (float(np.average(yy[win], weights=ww[win]))
               if win.any() else np.nan)
    avg_loss = (float(np.average((-yy)[~win], weights=ww[~win]))
                if (~win).any() else np.nan)
    return {
        "n_rows": int(mask.sum()),
        "mean_predicted_pwin": float(np.average(p_hat[mask], weights=ww)),
        "mean_predicted_rr": float(np.average(rr[mask], weights=ww)),
        "actual_win_rate": float(np.average(win.astype(float), weights=ww)),
        "actual_avg_win": avg_win,
        "actual_avg_loss": avg_loss,
        "actual_payoff_ratio": (float(avg_win / avg_loss)
                                if (avg_loss and avg_loss > 0) else np.nan),
        "actual_mean_return_atr": float(np.average(yy, weights=ww)),
    }


def build_val_grid(p_hat, mu_w, mu_l, ev_c, y_ret, wgt):
    """Fixed 5x5 Pwin-quintile x RR-quintile grid + 10 combined-EV deciles."""
    rr = np.where(mu_l > 0, mu_w / mu_l, np.inf)
    qp = _safe_quintile(p_hat, 5)
    qr = _safe_quintile(np.where(np.isfinite(rr), rr, np.inf), 5)
    grid = []
    for a in range(5):
        for b in range(5):
            cell = _cell_stats((qp == a) & (qr == b), p_hat, rr, y_ret, wgt, None)
            if cell is not None:
                cell.update({"pwin_quintile": a + 1, "rr_quintile": b + 1})
                grid.append(cell)

    dec = _safe_quintile(ev_c, 10)
    deciles = []
    for d in range(10):
        cell = _cell_stats(dec == d, p_hat, rr, y_ret, wgt, None)
        if cell is not None:
            cell.update({"decile": d + 1})
            deciles.append(cell)
    return {"grid_5x5": grid, "ev_combined_deciles": deciles}


def val_combined_diagnostics(win_bundles, payoff_bundles, priors,
                             win_frames=None, payoff_frames=None):
    """VAL only. Combine R9A + R9B predictions via the Composer and build the
    descriptive 5x5 grid + EV deciles per horizon."""
    if win_frames is None:
        win_frames = win_mod.load_fit_frames()
    if payoff_frames is None:
        payoff_frames = pay_mod.load_fit_frames()
    out = {}
    for H in HORIZONS:
        wf = win_frames["val"]
        wf = wf[wf["horizon"] == H]
        pf = payoff_frames["val"]
        pf = pf[pf["horizon"] == H]
        p_hat = win_mod.predict_win_probability(
            win_bundles[H], wf[list(WIN33_COLS)].to_numpy(np.float32))
        mu_w, mu_l, _ = pay_mod.predict_payoff(
            payoff_bundles[H], pf[list(PAY8_COLS)].to_numpy(np.float32))
        pr = priors[H]
        ev_w, ev_r, ev_c = compose_scores(
            p_hat, mu_w, mu_l,
            p_train_prior=pr["p0"], mu_w_train_prior=pr["muW0"],
            mu_l_train_prior=pr["muL0"])
        y_ret = wf["episode_return_atr"].to_numpy(float)
        wgt = wf["sample_weight"].to_numpy(float)
        grid = build_val_grid(p_hat, mu_w, mu_l, ev_c, y_ret, wgt)
        out[H] = {
            "ev_w": {"mean": float(np.average(ev_w, weights=wgt)),
                     "positive_share": float(np.mean(ev_w > 0.0))},
            "ev_r": {"mean": float(np.average(ev_r, weights=wgt)),
                     "positive_share": float(np.mean(ev_r > 0.0))},
            "ev_c": {"mean": float(np.average(ev_c, weights=wgt)),
                     "positive_share": float(np.mean(ev_c > 0.0))},
            **grid,
        }
    return out


if __name__ == "__main__":
    priors = compute_train_priors()
    for H in HORIZONS:
        print(H, {k: (round(v, 4) if isinstance(v, float) else v)
                  for k, v in priors[H].items()})
