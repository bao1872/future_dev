"""PATH-0 — deterministic synthetic / contract tests.

覆盖：
  * path morphology 14 个公式的确定性数值
  * 只读 prev.start_bar..prev.end_bar（不越界到 target 之后）
  * contiguous pair 过滤语义
  * 四-bit 展开（不做 bullish/bearish 映射，不塌成 multiclass）
  * ECE 固定 10 bins
  * episode identity / sample key hash 排序不变性
  * 预注册 feature set 精确性

运行：
  .venv/bin/python research/liquidity_oracle_atlas/test_path0_episode_path_memory_v1.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import research.liquidity_oracle_atlas.experiment_path0_episode_path_memory_v1 as P  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name} :: {detail}")
        FAILS.append(name)


# --------------------------------------------------------------- path math
def test_path_morphology_formulas():
    # bars 0..4, episode start s=0, end e=4, A = 2.0, U = 12, D = 8 (W = 4)
    c = np.array([10.0, 11.0, 10.0, 11.0, 13.0])
    h = np.array([10.5, 11.5, 10.5, 11.5, 13.5])
    l = np.array([9.5, 10.5, 9.5, 10.5, 12.5])
    f = P.path_morphology(h, l, c, 2.0, 12.0, 8.0, 0, 4)
    d = 4
    # delta = [+1, -1, +1, +2] -> tv = 5, net = 3
    check("net_move_R", abs(f["prev_net_move_R"] - 1.5) < 1e-12,
          f["prev_net_move_R"])
    check("total_variation_R", abs(f["prev_total_variation_R"] - 2.5) < 1e-12,
          f["prev_total_variation_R"])
    check("signed_efficiency", abs(f["prev_signed_efficiency"] - 0.6) < 1e-12,
          f["prev_signed_efficiency"])
    # path_high = max(close[0]=10, max h[1..4]=13.5) = 13.5 ; low = min(10, 9.5) = 9.5
    check("range_R", abs(f["prev_range_R"] - (13.5 - 9.5) / 2.0) < 1e-12,
          f["prev_range_R"])
    check("max_up_excursion_R",
          abs(f["prev_max_up_excursion_R"] - (13.5 - 10.0) / 2.0) < 1e-12,
          f["prev_max_up_excursion_R"])
    check("max_down_excursion_R",
          abs(f["prev_max_down_excursion_R"] - (10.0 - 9.5) / 2.0) < 1e-12,
          f["prev_max_down_excursion_R"])
    # nonzero delta = all 4 ; sign = + - + + -> 2 flips / 3
    check("direction_change_rate",
          abs(f["prev_direction_change_rate"] - 2.0 / 3.0) < 1e-12,
          f["prev_direction_change_rate"])
    # d=4 -> k = ceil(4/2) = 2 -> mid = 2
    check("first_half_net_R",
          abs(f["prev_first_half_net_R"] - (10.0 - 10.0) / 2.0) < 1e-12,
          f["prev_first_half_net_R"])
    check("second_half_net_R",
          abs(f["prev_second_half_net_R"] - (13.0 - 10.0) / 2.0) < 1e-12,
          f["prev_second_half_net_R"])
    check("close_location_in_range",
          abs(f["prev_close_location_in_range"] - (13.0 - 9.5) / 4.0) < 1e-12,
          f["prev_close_location_in_range"])
    # min((U - h)/W) over h[1..4] = min((12-11.5),(12-10.5),(12-11.5),(12-13.5))/4
    check("min_upper_high_gap_frac (negative allowed)",
          abs(f["prev_min_upper_high_gap_frac"] - (-1.5 / 4.0)) < 1e-12,
          f["prev_min_upper_high_gap_frac"])
    check("min_lower_low_gap_frac",
          abs(f["prev_min_lower_low_gap_frac"] - (1.5 / 4.0)) < 1e-12,
          f["prev_min_lower_low_gap_frac"])
    check("upper_touch_rate", abs(f["prev_upper_touch_rate"] - 0.0) < 1e-12,
          f["prev_upper_touch_rate"])
    check("lower_touch_rate", abs(f["prev_lower_touch_rate"] - 0.0) < 1e-12,
          f["prev_lower_touch_rate"])
    check("all 14 features produced", len(f) == 14, len(f))


def test_exact_touch_rate_and_degenerate():
    c = np.array([10.0, 10.0, 10.0])
    h = np.array([10.5, 12.0, 10.5])
    l = np.array([9.5, 10.0, 9.5])
    f = P.path_morphology(h, l, c, 1.0, 12.0, 8.0, 0, 2)
    check("upper_touch_rate exact equality counts",
          abs(f["prev_upper_touch_rate"] - 0.5) < 1e-12,
          f["prev_upper_touch_rate"])
    g = P.path_morphology(h, l, c, 1.0, 12.0, 10.0, 0, 2)
    check("both touch rates count exact equality (U=12.0, D=10.0)",
          abs(g["prev_lower_touch_rate"] - 0.5) < 1e-12
          and abs(g["prev_upper_touch_rate"] - 0.5) < 1e-12,
          (g["prev_upper_touch_rate"], g["prev_lower_touch_rate"]))

    # flat path: path_high == path_low -> 0.5, and dc=0
    c2 = np.array([10.0, 10.0, 10.0])
    f2 = P.path_morphology(c2 + 0.5, c2 - 0.5, c2, 1.0, 12.0, 8.0, 0, 2)
    check("flat path close_location_in_range == 0.5",
          abs(f2["prev_close_location_in_range"] - 0.5) < 1e-12,
          f2["prev_close_location_in_range"]
          if (f2["prev_close_location_in_range"] != 0.5) else "")
    check("no nonzero delta -> direction_change_rate 0",
          f2["prev_direction_change_rate"] == 0.0,
          f2["prev_direction_change_rate"])


def test_no_future_bars_read():
    """把 target 之后的 bar 改成异常值，path features 必须完全不变。"""
    c = np.array([10.0, 11.0, 10.0, 11.0, 13.0, 999.0])
    h = np.array([10.5, 11.5, 10.5, 11.5, 13.5, 999.0])
    l = np.array([9.5, 10.5, 9.5, 10.5, 12.5, 0.0])
    f = P.path_morphology(h, l, c, 2.0, 12.0, 8.0, 0, 4)
    check("bar e+1 never enters path features",
          abs(f["prev_net_move_R"] - 1.5) < 1e-12
          and abs(f["prev_range_R"] - 2.0) < 1e-12, f["prev_range_R"])


# ------------------------------------------------------- contiguous pairs
def test_contiguous_pair_filter():
    ep = pd.DataFrame([
        dict(symbol="X", start_bar=0, end_bar=3, gap_bars_before_episode=0,
             gap_bars_after_episode=0),
        dict(symbol="X", start_bar=3, end_bar=5, gap_bars_before_episode=0,
             gap_bars_after_episode=2),
        dict(symbol="X", start_bar=7, end_bar=9, gap_bars_before_episode=2,
             gap_bars_after_episode=0),
        dict(symbol="Y", start_bar=0, end_bar=1, gap_bars_before_episode=0,
             gap_bars_after_episode=0),
    ])
    pairs = []
    for sym, g in ep.groupby("symbol", sort=False):
        g = g.reset_index(drop=True)
        sb = g["start_bar"].to_numpy()
        eb = g["end_bar"].to_numpy()
        gaf = g["gap_bars_after_episode"].to_numpy()
        gbf = g["gap_bars_before_episode"].to_numpy()
        ok = (eb[:-1] == sb[1:]) & (gaf[:-1] == 0) & (gbf[1:] == 0)
        pairs += [(sym, int(a), int(b)) for a, b in
                  zip(np.flatnonzero(ok), np.flatnonzero(ok) + 1)]
    check("only the gap-free contiguous pair survives",
          pairs == [("X", 0, 1)], pairs)


# ----------------------------------------------------------------- labels
def test_four_bit_labels():
    masks = np.array([1, 2, 3, 4, 8, 10, 15])
    Y = np.stack([((masks & m) != 0).astype(int) for m in P.BIT_MASKS], axis=1)
    check("bit order is UP,DOWN,NEW_UPPER,NEW_LOWER",
          P.BIT_NAMES == ["UP_PEN", "DOWN_PEN", "NEW_UPPER", "NEW_LOWER"],
          P.BIT_NAMES)
    check("mask 10 -> DOWN_PEN + NEW_LOWER",
          Y[5].tolist() == [0, 1, 0, 1], Y[5].tolist())
    check("mask 15 -> all four (no multiclass collapse)",
          Y[6].sum() == 4, Y[6].tolist())
    check("no bullish/bearish remapping exists",
          not hasattr(P, "SIGNED_MASK") and not hasattr(P, "BULLISH"), "")


# -------------------------------------------------------------------- ECE
def test_ece_fixed_bins():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.05, 0.05, 0.95, 0.95])
    e = P.ece_binary(y, p)
    # bin0: mean p .05, mean y 0 -> .5*|.05-0| ; bin9: .5*|.95-1|
    check("ECE fixed 10 bins value",
          abs(e - (0.5 * 0.05 + 0.5 * 0.05)) < 1e-12, e)
    perfect = P.ece_binary(np.array([0, 1]), np.array([0.0, 1.0]))
    check("ECE clips boundary probabilities without error",
          abs(perfect - 0.0) < 1e-12, perfect)


def test_logloss_brier():
    y = np.array([0, 1])
    p = np.array([0.0, 1.0])
    check("logloss perfect == 0",
          abs(P.binary_logloss(y, p)) < 1e-12, P.binary_logloss(y, p))
    check("brier perfect == 0", abs(P.binary_brier(y, p)) < 1e-12,
          P.binary_brier(y, p))
    check("logloss worse than brier scale sanity",
          P.binary_logloss(y, np.array([0.5, 0.5])) > 0.69, "")


# ------------------------------------------------------------------ hashes
def test_hash_sort_invariance():
    a = pd.DataFrame([
        dict(symbol="X", start_bar=1, end_bar=2, event_mask=1),
        dict(symbol="Y", start_bar=1, end_bar=3, event_mask=4)])
    b = a.iloc[::-1].reset_index(drop=True)
    check("episode identity hash is order invariant",
          P.episode_identity_hash(a) == P.episode_identity_hash(b), "")


# -------------------------------------------------------- registered spec
def test_registered_feature_sets():
    check("14 path features exactly as preregistered",
          P.PATH_NUM == [
              "prev_net_move_R", "prev_total_variation_R",
              "prev_signed_efficiency", "prev_range_R",
              "prev_max_up_excursion_R", "prev_max_down_excursion_R",
              "prev_direction_change_rate", "prev_first_half_net_R",
              "prev_second_half_net_R", "prev_close_location_in_range",
              "prev_min_upper_high_gap_frac", "prev_min_lower_low_gap_frac",
              "prev_upper_touch_rate", "prev_lower_touch_rate"],
          P.PATH_NUM)
    check("M0 has no history", P.MODELS[P.M0][1] == []
          and P.MODELS[P.M0][0] == P.CUR_NUM, P.MODELS[P.M0])
    check("M1 = M0 + prev_event only",
          P.MODELS[P.M1][0] == P.CUR_NUM
          and P.MODELS[P.M1][1] == ["prev_event_mask"], P.MODELS[P.M1])
    check("M2 = M1 + duration only",
          P.MODELS[P.M2][0] == P.CUR_NUM + ["prev_duration_bars"],
          P.MODELS[P.M2])
    check("M3 = M2 + path exactly",
          P.MODELS[P.M3][0] == P.MODELS[P.M2][0] + P.PATH_NUM,
          len(P.MODELS[P.M3][0]))
    banned = ("symbol", "type", "scope", "volume", "return", "signature",
              "smc", "bos", "choch", "sweep")
    bad_feat = [c for v in P.MODELS.values() for c in (v[0] + v[1])
                if any(b in c.lower() for b in banned)]
    check("no symbol / type / scope / volume / return / SMC / signature "
          "feature anywhere", bad_feat == [], bad_feat)
    check("primary comparison is M3 - M2",
          P.COMPARISONS[0][0] == P.M3 and P.COMPARISONS[0][1] == P.M2,
          P.COMPARISONS[0][:2])
    check("bootstrap seed / reps frozen",
          P.BOOTSTRAP_SEED == 20260914 and P.BOOTSTRAP_REPS == 1000,
          (P.BOOTSTRAP_SEED, P.BOOTSTRAP_REPS))
    check("frozen episode hash constant present",
          len(P.REVIEWER_FROZEN_EPISODE_HASH) == 64, "")


def test_pipeline_constructs():
    pipe = P.make_pipeline(P.CUR_NUM + P.M2_EXTRA_NUM, P.PREV_CAT)
    X = pd.DataFrame({**{c: [1.0, 2.0, 3.0, 4.0] for c in P.CUR_NUM},
                      "prev_duration_bars": [1, 2, 3, 4],
                      "prev_event_mask": [1, 3, 4, 15]})
    y = np.array([0, 1, 0, 1])
    pipe.fit(X, y)
    p = pipe.predict_proba(X)[:, 1]
    check("M2 pipeline fits and predicts in (0,1)",
          p.shape == (4,) and ((p > 0) & (p < 1)).all(), p)


def main():
    test_path_morphology_formulas()
    test_exact_touch_rate_and_degenerate()
    test_no_future_bars_read()
    test_contiguous_pair_filter()
    test_four_bit_labels()
    test_ece_fixed_bins()
    test_logloss_brier()
    test_hash_sort_invariance()
    test_registered_feature_sets()
    test_pipeline_constructs()
    print(f"\n==== {len(FAILS)} FAIL ====")
    if FAILS:
        print("FAILED:", FAILS)
        sys.exit(1)
    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
