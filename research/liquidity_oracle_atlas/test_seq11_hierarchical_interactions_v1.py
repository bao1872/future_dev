"""SEQ-1.1 — deterministic synthetic / contract tests.

覆盖：
  * 16 个 ordered interaction 的定义（顺序不可交换，correct AND semantics）
  * R0–R3 的嵌套结构：R3 必须保留两个 main effects（reviewer correction）
  * R0/R1/R2 与 SEQ-1 Q0/Q1/Q2 的 feature set 等价性
  * window / seed / comparison 集合冻结；primary = R3 - R2
  * interaction sparsity：不删除、不新增、16 个固定
  * coefficient 提取的列顺序安全（pipeline layout guard）
  * 运行时：cross-check、outputs、TB4 措辞、无原始 bar 读取

运行：
  .venv/bin/python research/liquidity_oracle_atlas/test_seq11_hierarchical_interactions_v1.py
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import research.liquidity_oracle_atlas.experiment_seq11_hierarchical_interactions_v1 as X  # noqa: E402
import research.liquidity_oracle_atlas.experiment_seq1_second_order_grammar_v1 as G  # noqa: E402
import research.liquidity_oracle_atlas.experiment_state11_functional_provenance_v1 as T  # noqa: E402
import research.liquidity_oracle_atlas.experiment_state1_provenance_decomposition_v1 as S  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name} :: {detail}")
        FAILS.append(name)


# ------------------------------------------------------------ interactions
def test_interaction_names_and_order():
    check("exactly 16 fixed interactions", len(X.INTERACTIONS) == 16,
          X.INTERACTIONS)
    check("names follow pp_<i>_x_p_<j> in the preregistered order",
          X.INTERACTIONS[:4] == ["pp_UP_PEN_x_p_UP_PEN",
                                 "pp_UP_PEN_x_p_DOWN_PEN",
                                 "pp_UP_PEN_x_p_NEW_UPPER",
                                 "pp_UP_PEN_x_p_NEW_LOWER"]
          and X.INTERACTIONS[4:8] == ["pp_DOWN_PEN_x_p_UP_PEN",
                                     "pp_DOWN_PEN_x_p_DOWN_PEN",
                                     "pp_DOWN_PEN_x_p_NEW_UPPER",
                                     "pp_DOWN_PEN_x_p_NEW_LOWER"],
          X.INTERACTIONS[:8])
    check("all 16 names are unique", len(set(X.INTERACTIONS)) == 16, "")
    check("order is non-commutative: pp_A_x_p_B != pp_B_x_p_A",
          "pp_UP_PEN_x_p_NEW_LOWER" in X.INTERACTIONS
          and "pp_NEW_LOWER_x_p_UP_PEN" in X.INTERACTIONS
          and "pp_UP_PEN_x_p_NEW_LOWER"
          != "pp_NEW_LOWER_x_p_UP_PEN", "")
    check("bit definitions are the four primitive structural bits",
          X.BIT_DEFS == [("UP_PEN", 1), ("DOWN_PEN", 2),
                         ("NEW_UPPER", 4), ("NEW_LOWER", 8)], X.BIT_DEFS)


def test_interaction_values():
    df = pd.DataFrame(dict(
        prevprev_event_mask=[1, 1, 3, 12, 0, 9],
        prev_event_mask=[2, 1, 8, 12, 4, 4]))
    df = X.add_interactions(df)
    check("all 16 columns created",
          all(c in df.columns for c in X.INTERACTIONS), list(df.columns))
    check("pp_UP_PEN_x_p_DOWN_PEN = AND semantics",
          df["pp_UP_PEN_x_p_DOWN_PEN"].tolist() == [1., 0., 0., 0., 0., 0.],
          df["pp_UP_PEN_x_p_DOWN_PEN"].tolist())
    # (pp,p): (1,2)->0 (1,1)->1 (3,8)->0 (12,12)->0 (0,4)->0 (9,4)->0
    check("pp_UP_PEN_x_p_UP_PEN fired only when both sides have UP_PEN",
          df["pp_UP_PEN_x_p_UP_PEN"].tolist() == [0., 1., 0., 0., 0., 0.],
          df["pp_UP_PEN_x_p_UP_PEN"].tolist())
    check("pp_DOWN_PEN_x_p_NEW_LOWER (prevprev DOWN_PEN & prev NEW_LOWER)",
          df["pp_DOWN_PEN_x_p_NEW_LOWER"].tolist()
          == [0., 0., 1., 0., 0., 0.],
          df["pp_DOWN_PEN_x_p_NEW_LOWER"].tolist())
    check("pp_NEW_UPPER_x_p_NEW_UPPER on masks (12,12)",
          df["pp_NEW_UPPER_x_p_NEW_UPPER"].tolist()
          == [0., 0., 0., 1., 0., 0.],
          df["pp_NEW_UPPER_x_p_NEW_UPPER"].tolist())
    check("interactions are 0/1 numeric",
          all(set(df[c].unique()) <= {0.0, 1.0} for c in X.INTERACTIONS), "")


# ----------------------------------------------------------------- models
def test_nested_model_structure():
    check("R0 = current state only",
          X.MODELS[X.R0] == (X.CURRENT, []), X.MODELS[X.R0])
    check("CURRENT = 4 geometry + 10 functional provenance",
          X.CURRENT == list(X.GEOM) + list(X.FUNC_PROV)
          and len(X.CURRENT) == 14, len(X.CURRENT))
    check("R1 = R0 + prev_event_mask only",
          X.MODELS[X.R1] == (X.CURRENT, ["prev_event_mask"]),
          X.MODELS[X.R1])
    check("R2 = R1 + prevprev_event_mask",
          X.MODELS[X.R2] == (X.CURRENT,
                             ["prev_event_mask", "prevprev_event_mask"]),
          X.MODELS[X.R2])
    check("R3 = R2 + 16 interactions, BOTH main effects retained",
          X.MODELS[X.R3] == (X.CURRENT + X.INTERACTIONS,
                             ["prev_event_mask", "prevprev_event_mask"]),
          X.MODELS[X.R3])
    check("reviewer correction honoured: R3 is not pair-only",
          "prevprev_event_mask" in X.MODELS[X.R3][1]
          and "prev_event_mask" in X.MODELS[X.R3][1], X.MODELS[X.R3][1])
    check("R3 numeric block is exactly current state + 16 interactions",
          X.MODELS[X.R3][0] == X.CURRENT + X.INTERACTIONS, "")
    check("R1 numeric/categorical identical to SEQ-1 Q1",
          X.MODELS[X.R1] == (T.MODELS[T.F1][0], ["prev_event_mask"]), "")
    check("R1/R2 equivalent to SEQ-1 Q1/Q2",
          X.MODELS[X.R1] == (G.MODELS[G.Q1][0], ["prev_event_mask"])
          and X.MODELS[X.R2] == (G.MODELS[G.Q2][0],
                                 ["prev_event_mask", "prevprev_event_mask"]),
          (X.MODELS[X.R1], X.MODELS[X.R2]))
    banned = ("duration", "path", "symbol", "type", "scope", "volume",
              "signature", "smc", "bos", "choch", "sweep", "hmm",
              "prevprevprev", "third_order", "same_side", "opposite_side")
    bad = [c for v in X.MODELS.values() for c in (v[0] + v[1])
           if any(b in c.lower() for b in banned)]
    check("no duration / path / symbol / SMC / third-order / hand-made role",
          bad == [], bad)
    check("primary comparison is R3 - R2",
          X.COMPARISONS[0] == (X.R3, X.R2), X.COMPARISONS[0])
    check("exactly 4 frozen comparisons",
          len(X.COMPARISONS) == 4
          and set(X.COMPARISONS) == {(X.R3, X.R2), (X.R2, X.R1),
                                     (X.R1, X.R0), (X.R3, X.R1)},
          X.COMPARISONS)


def test_windows_and_frozen_constants():
    w = X.WINDOWS
    check("window A = TB1 -> TB2 seed 20260920",
          w[0]["train"] == ["TB1"] and w[0]["eval"] == "TB2"
          and w[0]["seed"] == 20260920, w[0])
    check("window B = TB1+TB2 -> TB3 seed 20260921",
          w[1]["train"] == ["TB1", "TB2"] and w[1]["eval"] == "TB3"
          and w[1]["seed"] == 20260921, w[1])
    check("no window touches TB4",
          all("TB4" not in r["train"] and r["eval"] != "TB4" for r in w), w)
    check("frozen seq1 sample hash constant is the reviewer value",
          X.FROZEN_SEQ1_HASH
          == "7b28e3aa9a41a0b27e400cf9bb060f1048a8e3ebf95e0e33780855f190989067",
          X.FROZEN_SEQ1_HASH)
    check("hash function is SEQ-1's (same object)", X.triple_hash is G.triple_hash,
          "")
    check("bootstrap reps frozen at 1000", X.BOOTSTRAP_REPS == 1000, "")


def test_no_raw_bar_loader():
    src = Path(X.__file__).read_text()
    check("script never imports load_raw_bars",
          "load_raw_bars" not in src, "")
    check("script never imports build_blocks",
          "build_blocks" not in src, "")
    check("script never claims 'TB4 never read'",
          "TB4 never read" not in src and "tb4_read_for_values" not in src, "")
    check("script records tb4_analytically_used",
          "tb4_analytically_used" in src, "")


def test_coefficient_extraction_layout():
    """系数必须确实对应 16 个 interaction 列（不是别的列）。"""
    rng = np.random.default_rng(11)
    n = 400
    df = pd.DataFrame({c: rng.normal(size=n) for c in X.CURRENT})
    df["prevprev_event_mask"] = rng.integers(0, 16, n)
    df["prev_event_mask"] = rng.integers(1, 16, n)
    df = X.add_interactions(df)
    num_cols, cat_cols = X.MODELS[X.R3]
    pipe = G.make_pipeline(num_cols, cat_cols)
    y = (df["pp_UP_PEN_x_p_UP_PEN"].to_numpy() * 0
         + (df["prev_event_mask"].to_numpy() == 1).astype(int))
    pipe.fit(df[num_cols + cat_cols], y)
    coef = X.interaction_coefficients(pipe, num_cols)
    coef_all = pipe.named_steps["clf"].coef_.ravel()
    check("16 coefficients returned", coef.shape == (16,), coef.shape)
    check("coefficients are the LAST 16 numeric-block entries",
          np.allclose(coef, coef_all[len(num_cols) - 16:len(num_cols)]), "")
    check("layout guard rejects a mismatched column list",
          _raises(lambda: X.interaction_coefficients(pipe, num_cols[:-1])), "")


def _raises(fn):
    try:
        fn()
        return False
    except SystemExit:
        return True


def test_outputs_if_present():
    f = X.OUT / "seq11_summary.json"
    if not f.exists():
        check("seq11_summary.json exists (skipped)", True, "")
        return
    d = json.loads(f.read_text())
    cc = d["frozen_crosscheck"]
    check("frozen cross-check exact on both windows",
          all(v["exact"] is True for v in cc.values()), cc)
    check("exactly 4 comparisons per window (8 rows)",
          len(d["bootstrap"]) == 8, len(d["bootstrap"]))
    check("tb4_analytically_used is false",
          d["tb4_analytically_used"] is False, d["tb4_analytically_used"])
    check("interaction policy states never-delete/never-add",
          "never deleted or added" in d["interactions"]["policy"], "")
    sp = pd.read_csv(X.OUT / "seq11_interaction_sparsity.csv")
    check("sparsity table has 16 rows per window (32 total)",
          len(sp) == 32 and sp["window"].nunique() == 2, len(sp))
    check("all 16 interactions retained with n_positive > 0",
          bool((sp["n_positive"] > 0).all()), "")
    co = pd.read_csv(X.OUT / "seq11_interaction_coefficients.csv")
    check("coefficient table has 16 rows x 4 bits x 2 windows",
          len(co) == 128 and co["bit"].nunique() == 4
          and co["window"].nunique() == 2, len(co))
    check("coefficient rows cover exactly the 16 fixed interactions",
          set(co["interaction"]) == set(X.INTERACTIONS), "")
    for nm in ["seq11_model_metrics.csv", "seq11_bootstrap.csv",
               "seq11_per_bit.csv", "seq11_by_symbol.csv"]:
        check(f"{nm} written", (X.OUT / nm).exists(), nm)


def main():
    test_interaction_names_and_order()
    test_interaction_values()
    test_nested_model_structure()
    test_windows_and_frozen_constants()
    test_no_raw_bar_loader()
    test_coefficient_extraction_layout()
    test_outputs_if_present()
    print(f"\n==== {len(FAILS)} FAIL ====")
    if FAILS:
        print("FAILED:", FAILS)
        sys.exit(1)
    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
