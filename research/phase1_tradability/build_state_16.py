"""为 16 品种重建 ob_rl_state_v0（105 维事件状态），不触碰 DEV4 冻结产物。

复用 build_ob_rl_dataset_v0 的全部特征语义，只做两处重定向：
  1. OUT_ROOT      -> ob_rl_dataset_v0_16（避免覆盖 DEV4 的 state/action）
  2. load_full_or_chunks -> 读取 analysis_data 下「全部品种」的 per-symbol
     chunks（DEV4 四个 + 本轮新增十二个），而不是只含 DEV4 的合并 CSV。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import research.build_ob_rl_dataset_v0 as B

GIT = Path("research/analysis_data/ob_candidate_universe_v3")
OUT16 = Path("research/analysis_results/ob_rl_dataset_v0_16")


def load_all_symbols(table: str) -> pd.DataFrame:
    paths = sorted((GIT / table).glob("*/*.csv"))
    if not paths:
        raise FileNotFoundError(f"{table}: no per-symbol chunks")
    syms = sorted({p.parent.name for p in paths})
    print(f"[state16] {table}: {len(paths)} chunks, symbols={syms}",
          flush=True)
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def main():
    OUT16.mkdir(parents=True, exist_ok=True)
    B.OUT_ROOT = OUT16
    B.load_full_or_chunks = load_all_symbols
    B.main()

    csv = OUT16 / "ob_rl_state_v0.csv"
    st = pd.read_csv(csv, low_memory=False)
    st.to_parquet(OUT16 / "ob_rl_state_v0.parquet", index=False)
    print(f"[state16] rows={len(st)} "
          f"symbols={sorted(st['symbol'].unique().tolist())}")
    print("STATE16_DONE")


if __name__ == "__main__":
    main()
