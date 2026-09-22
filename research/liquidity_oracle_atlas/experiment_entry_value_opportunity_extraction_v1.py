"""M1 Opportunity Extraction V1.

Lightweight selection analysis on top of the ALREADY-CACHED Phase-2 M1/STRUCT44
validation/test predictions. It does NOT rebuild the Oracle, does NOT retrain
LightGBM, does NOT modify the cost kernel, and adds NO new features.

Scientific question
-------------------
Can the frozen M1/STRUCT44 prediction scores be thresholded into a subset whose
causal first-entry opportunity has mean actual Oracle Entry Value > 0 out of
sample?

The score is, per candidate row:

    score      = max(pL, pS)
    direction  = LONG  if pL >= pS  else SHORT
    actual_y   = Y_L   if pL >= pS  else Y_S

where pL/pS are the M1 Long/Short validation/test predictions.

Experimental discipline (frozen)
-------------------------------
* Coverages: 20%, 10%, 5%, 3%, 2%, 1%, 0.5%.
* For each kappa and coverage q, the numeric threshold tau is derived from
  VALIDATION only: tau = quantile(validation_score, 1 - q, method="higher").
* That exact numeric tau is then applied, unchanged, to TEST.
* The Primary causal execution diagnostic is FIRST CROSSING per episode: each
  proximity episode contributes exactly one selected row -- the earliest bar at
  which score >= tau. No future-best selection.
* Validation threshold selection scans coverages from largest to smallest and
  picks the FIRST one whose validation first-crossing has >= 100 episodes,
  mean actual_y > 0 and bootstrap 95% CI lower bound > 0.
* Test is evaluated exactly ONCE, with the validation-selected tau.
* Bootstrap: 1000 episode resamples of mean(actual_y), seed 20260921.

Verdict classification is the reviewer's job; this module only records the
inputs.

Outputs (4 small artifacts, no parquet, no models):
    artifacts/entry_value_opportunity_extraction_v1/
        opportunity_grid.csv
        opportunity_selected_policy.csv
        opportunity_symbol_test.csv
        opportunity_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# frozen configuration
# --------------------------------------------------------------------------- #
TASK_ID = "FUTURE-ENTRY-VALUE-M1-OPPORTUNITY-EXTRACTION-V1"
BASE_SHA = "2d056ec871d599c1ba4a55133715c9cdaec77d91"
PARENT_SHA = BASE_SHA  # continue on the same branch

PRED_DIR = Path(
    "artifacts/entry_value_cost_robustness_m1_v1/phase2_cache")
OUT_DIR = Path("artifacts/entry_value_opportunity_extraction_v1")

KAPPAS = [0.0, 0.005, 0.01, 0.02, 0.05]
COVERAGES = (0.20, 0.10, 0.05, 0.03, 0.02, 0.01, 0.005)
MIN_EPISODES = 100
BOOTSTRAP_N = 1000
SEED = 20260921

PHASE2_SUMMARY = Path(
    "artifacts/entry_value_cost_robustness_m1_v1/cost_summary.json")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_prediction_cache(kappa: float) -> pd.DataFrame:
    """Read the cached Phase-2 M1 prediction parquet for a kappa."""
    path = PRED_DIR / f"pred_{kappa:.4f}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"STOP_OPPORTUNITY_PRED_CACHE_MISSING: {path}")
    return pd.read_parquet(path)


def _score_cols(split: str) -> tuple[str, str]:
    if split == "validation":
        return "p_va_long_M1", "p_va_short_M1"
    if split == "test":
        return "p_te_long_M1", "p_te_short_M1"
    raise ValueError(f"split must be validation/test, got {split!r}")


def build_score_frame(pred: pd.DataFrame, split: str) -> pd.DataFrame:
    """Attach score / direction / actual_y to the rows of one split."""
    d = pred[pred["split"] == split].copy()
    pL_col, pS_col = _score_cols(split)
    pL = d[pL_col].to_numpy(float)
    pS = d[pS_col].to_numpy(float)
    assert np.isfinite(pL).all(), f"non-finite {pL_col} in {split}"
    assert np.isfinite(pS).all(), f"non-finite {pS_col} in {split}"

    choose_long = pL >= pS
    yl = d["Y_L"].to_numpy(float)
    ys = d["Y_S"].to_numpy(float)

    d["score"] = np.maximum(pL, pS)
    d["direction"] = np.where(choose_long, "LONG", "SHORT")
    d["actual_y"] = np.where(choose_long, yl, ys)
    assert np.isfinite(d["actual_y"].to_numpy(float)).all(), \
        "non-finite actual_y"
    return d


def threshold_from_validation(val: pd.DataFrame, coverage: float) -> float:
    """tau = (1 - coverage) quantile of validation score, method='higher'."""
    return float(np.quantile(
        val["score"].to_numpy(float), 1.0 - coverage, method="higher"))


def first_crossing(df: pd.DataFrame, tau: float) -> pd.DataFrame:
    """Keep only the earliest score>=tau row per (symbol, global_episode)."""
    selected = (
        df[df["score"] >= tau]
        .sort_values(["symbol", "global_episode", "decision_time"])
        .drop_duplicates(["symbol", "global_episode"], keep="first")
        .reset_index(drop=True)
    )
    assert not selected.duplicated(
        ["symbol", "global_episode"]).any(), \
        "first_crossing left duplicate episodes"
    assert np.isfinite(selected["actual_y"].to_numpy(float)).all()
    return selected


def bootstrap_mean_y(y: np.ndarray, n: int = BOOTSTRAP_N, seed: int = SEED):
    y = np.asarray(y, float)
    if len(y) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n, dtype=float)
    for i in range(n):
        sample = rng.choice(y, size=len(y), replace=True)
        means[i] = sample.mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def evaluate_first_crossing(
    sel: pd.DataFrame,
    total_rows: int,
    total_episodes: int,
) -> dict:
    """Primary metric block for a first-crossing selection."""
    n = len(sel)
    if n == 0:
        return {
            "selected_episodes": 0,
            "selected_symbols": 0,
            "row_coverage": 0.0,
            "episode_coverage": 0.0,
            "mean_actual_y": float("nan"),
            "median_actual_y": float("nan"),
            "positive_y_rate": float("nan"),
            "p25_actual_y": float("nan"),
            "p75_actual_y": float("nan"),
            "long_fraction": float("nan"),
            "short_fraction": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
        }
    y = sel["actual_y"].to_numpy(float)
    ci = bootstrap_mean_y(y)
    return {
        "selected_episodes": n,
        "selected_symbols": int(sel["symbol"].nunique()),
        "row_coverage": n / total_rows if total_rows else float("nan"),
        "episode_coverage": n / total_episodes if total_episodes else float("nan"),
        "mean_actual_y": float(np.mean(y)),
        "median_actual_y": float(np.median(y)),
        "positive_y_rate": float(np.mean(y > 0)),
        "p25_actual_y": float(np.percentile(y, 25)),
        "p75_actual_y": float(np.percentile(y, 75)),
        "long_fraction": float(np.mean(sel["direction"].to_numpy() == "LONG")),
        "short_fraction": float(np.mean(sel["direction"].to_numpy() == "SHORT")),
        "ci_lower": ci[0],
        "ci_upper": ci[1],
    }


def evaluate_row_pool(df: pd.DataFrame, tau: float) -> dict:
    """Diagnostic only: raw rows with score >= tau (no episode collapse)."""
    sel = df[df["score"] >= tau]
    if len(sel) == 0:
        return {"rp_rows": 0, "rp_mean_actual_y": float("nan"),
                "rp_positive_y_rate": float("nan")}
    y = sel["actual_y"].to_numpy(float)
    return {
        "rp_rows": len(sel),
        "rp_mean_actual_y": float(np.mean(y)),
        "rp_positive_y_rate": float(np.mean(y > 0)),
    }


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run_opportunity_extraction() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- hard check 1: all 5 prediction caches present ----
    missing = [k for k in KAPPAS
               if not (PRED_DIR / f"pred_{k:.4f}.parquet").exists()]
    if missing:
        raise FileNotFoundError(
            "STOP_OPPORTUNITY_PRED_CACHE_MISSING: " + str(missing))

    # lightgbm version + kappas from the Phase-2 summary (no model import)
    lgb_version = "unknown"
    if PHASE2_SUMMARY.exists():
        s = json.loads(PHASE2_SUMMARY.read_text())
        lgb_version = s.get("lightgbm_version", "unknown")

    grid_rows: list[dict] = []
    policy_rows: list[dict] = []
    symbol_rows: list[dict] = []

    per_kappa: dict[str, dict] = {}

    for k in KAPPAS:
        pred = load_prediction_cache(k)
        val = build_score_frame(pred, "validation")
        test = build_score_frame(pred, "test")
        val_rows = len(val)
        val_eps = int(val["global_episode"].nunique())
        test_rows = len(test)
        test_eps = int(test["global_episode"].nunique())

        selected_coverage = None
        selected_tau = None
        val_sel = None
        test_sel = None
        test_sel_symbols = None

        for q in COVERAGES:
            tau = threshold_from_validation(val, q)

            fc_val = first_crossing(val, tau)
            mv = evaluate_first_crossing(fc_val, val_rows, val_eps)
            rp_val = evaluate_row_pool(val, tau)

            fc_test = first_crossing(test, tau)
            mt = evaluate_first_crossing(fc_test, test_rows, test_eps)
            rp_test = evaluate_row_pool(test, tau)

            grid_rows.append({
                "kappa": k, "coverage": q, "split": "validation",
                "threshold": tau,
                **{f"fc_{c}": mv[c] for c in (
                    "selected_episodes", "selected_symbols", "row_coverage",
                    "episode_coverage", "mean_actual_y", "median_actual_y",
                    "positive_y_rate", "p25_actual_y", "p75_actual_y",
                    "long_fraction", "short_fraction", "ci_lower", "ci_upper")},
                "rp_rows": rp_val["rp_rows"],
                "rp_mean_actual_y": rp_val["rp_mean_actual_y"],
                "rp_positive_y_rate": rp_val["rp_positive_y_rate"],
            })
            grid_rows.append({
                "kappa": k, "coverage": q, "split": "test",
                "threshold": tau,
                **{f"fc_{c}": mt[c] for c in (
                    "selected_episodes", "selected_symbols", "row_coverage",
                    "episode_coverage", "mean_actual_y", "median_actual_y",
                    "positive_y_rate", "p25_actual_y", "p75_actual_y",
                    "long_fraction", "short_fraction", "ci_lower", "ci_upper")},
                "rp_rows": rp_test["rp_rows"],
                "rp_mean_actual_y": rp_test["rp_mean_actual_y"],
                "rp_positive_y_rate": rp_test["rp_positive_y_rate"],
            })

            # hard check 3: test never participates in threshold selection
            # (tau is derived from val only; test metrics are diagnostic here)
            if (selected_tau is None
                    and mv["selected_episodes"] >= MIN_EPISODES
                    and mv["mean_actual_y"] > 0
                    and mv["ci_lower"] > 0):
                selected_coverage = q
                selected_tau = tau
                val_sel = mv
                test_sel = mt
                test_sel_symbols = fc_test

        # ---- official Test evaluation uses ONLY the validation-selected tau ----
        if selected_tau is None:
            test_sel = {c: float("nan") for c in (
                "selected_episodes", "selected_symbols", "row_coverage",
                "episode_coverage", "mean_actual_y", "median_actual_y",
                "positive_y_rate", "p25_actual_y", "p75_actual_y",
                "long_fraction", "short_fraction", "ci_lower", "ci_upper")}
            test_sel_symbols = None

        # per-symbol Test evidence (selected threshold only)
        sym_positive_ge10 = None
        if test_sel_symbols is not None and len(test_sel_symbols):
            rep_syms = int(test_sel_symbols["symbol"].nunique())
            for sym, g in test_sel_symbols.groupby("symbol"):
                y = g["actual_y"].to_numpy(float)
                symbol_rows.append({
                    "kappa": k, "symbol": sym,
                    "selected_episodes": len(g),
                    "mean_actual_y": float(np.mean(y)),
                    "median_actual_y": float(np.median(y)),
                    "positive_y_rate": float(np.mean(y > 0)),
                    "long_count": int((g["direction"] == "LONG").sum()),
                    "short_count": int((g["direction"] == "SHORT").sum()),
                })
            ge10 = test_sel_symbols.groupby("symbol")["actual_y"].mean()
            ge10 = ge10[ge10.index.map(
                lambda s: (test_sel_symbols.groupby("symbol").size()[s] >= 10))]
            sym_positive_ge10 = int((ge10 > 0).sum())
        else:
            rep_syms = 0
            sym_positive_ge10 = None

        policy_rows.append({
            "kappa": k,
            "validation_selected_coverage": selected_coverage,
            "validation_threshold": selected_tau,
            "validation_episodes": (val_sel["selected_episodes"]
                                    if val_sel else 0),
            "validation_mean_y": (val_sel["mean_actual_y"]
                                  if val_sel else float("nan")),
            "validation_ci_lower": (val_sel["ci_lower"]
                                    if val_sel else float("nan")),
            "validation_ci_upper": (val_sel["ci_upper"]
                                    if val_sel else float("nan")),
            "test_episodes": test_sel["selected_episodes"],
            "test_actual_coverage": test_sel["episode_coverage"],
            "test_mean_y": test_sel["mean_actual_y"],
            "test_median_y": test_sel["median_actual_y"],
            "test_positive_y_rate": test_sel["positive_y_rate"],
            "test_ci_lower": test_sel["ci_lower"],
            "test_ci_upper": test_sel["ci_upper"],
            "test_long_fraction": test_sel["long_fraction"],
            "test_short_fraction": test_sel["short_fraction"],
            "test_symbols_represented": rep_syms,
            "test_symbols_positive_among_ge10": sym_positive_ge10,
        })

        per_kappa[str(k)] = {
            "validation_selected_coverage": selected_coverage,
            "validation_threshold": selected_tau,
            "validation_episodes": (val_sel["selected_episodes"]
                                    if val_sel else 0),
            "validation_mean_y": (val_sel["mean_actual_y"]
                                  if val_sel else None),
            "validation_ci_lower": (val_sel["ci_lower"]
                                    if val_sel else None),
            "test_episodes": test_sel["selected_episodes"],
            "test_mean_y": (None if np.isnan(test_sel["mean_actual_y"])
                            else test_sel["mean_actual_y"]),
            "test_ci_lower": (None if np.isnan(test_sel["ci_lower"])
                              else test_sel["ci_lower"]),
            "test_ci_upper": (None if np.isnan(test_sel["ci_upper"])
                              else test_sel["ci_upper"]),
            "test_positive_y_rate": test_sel["positive_y_rate"],
            "test_symbols_represented": rep_syms,
        }

    grid = pd.DataFrame(grid_rows)
    policy = pd.DataFrame(policy_rows)
    symbols = pd.DataFrame(symbol_rows)

    # hard check 4/5 already enforced inside first_crossing / build_score_frame
    hard_checks = {
        "all_prediction_caches_present": True,
        "thresholds_from_validation_only": True,
        "test_not_used_for_threshold_selection": True,
        "first_crossing_one_row_per_episode": True,
        "all_selected_actual_y_finite": True,
        "all_five_kappas_processed": len(per_kappa) == len(KAPPAS),
        "no_model_training": True,
        "no_oracle_call": True,
        "n_kappas": len(per_kappa),
        "lightgbm_version": lgb_version,
    }

    grid.to_csv(OUT_DIR / "opportunity_grid.csv", index=False)
    policy.to_csv(OUT_DIR / "opportunity_selected_policy.csv", index=False)
    symbols.to_csv(OUT_DIR / "opportunity_symbol_test.csv", index=False)

    summary = {
        "task_id": TASK_ID,
        "base_sha": BASE_SHA,
        "parent_sha": PARENT_SHA,
        "kappas": list(KAPPAS),
        "coverages": list(COVERAGES),
        "min_episodes": MIN_EPISODES,
        "bootstrap_n": BOOTSTRAP_N,
        "seed": SEED,
        "score_definition": "score=max(pL,pS); direction=argmax; actual_y=chosen Y",
        "primary_diagnostic": "first crossing per (symbol,global_episode)",
        "selection_rule": ("scan coverages 20%->0.5%, pick FIRST with "
                           "val episodes>=100 AND val mean_y>0 AND val ci_lower>0"),
        "lightgbm_version": lgb_version,
        "hard_checks": hard_checks,
        "verdict_inputs": per_kappa,
    }
    (OUT_DIR / "opportunity_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    print(f"[opportunity] grid rows={len(grid)} policy rows={len(policy)} "
          f"symbol rows={len(symbols)}")
    return summary


if __name__ == "__main__":
    run_opportunity_extraction()
