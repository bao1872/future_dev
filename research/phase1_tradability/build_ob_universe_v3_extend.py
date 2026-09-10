"""把 canonical OB candidate universe V3 从 DEV4 扩展到 16 品种。

严格复用 build_ob_candidate_universe_v3.process_symbol / write_git_chunks，
不复制、不重写任何 OB / SMC / DSA 语义，只是把它按品种增量地跑在
此前未参与开发的品种上。

用法（可分片并行）：
    python -m research.phase1_tradability.build_ob_universe_v3_extend \
        --symbols AL,AU,CF,I

产出：
    research/analysis_data/ob_candidate_universe_v3/{table}/{symbol}/*.csv
    research/exports/ob_candidate_universe_v3/{symbol}_{table}.csv

DEV4 既有产物完全不动。合并后的 candidates.csv 等由
merge_ob_universe_v3_16.py 单独重建。
"""
from __future__ import annotations

import argparse
import sys
import time

import research.build_ob_candidate_universe_v3 as v3


def run(symbols):
    for sym in symbols:
        t0 = time.perf_counter()
        res, stats = v3.process_symbol(sym)

        v3.write_git_chunks(res["candidates"], "candidates", sym)
        v3.write_git_chunks(res["context"], "context", sym)
        if not res["levels_compact"].empty:
            v3.write_git_chunks(res["levels_compact"], "levels", sym)
        v3.write_git_chunks(res["path"], "path", sym)
        v3.write_git_chunks(res["quantile"], "quantile", sym)

        (v3.LOCAL_OUT / f"{sym}_candidates.csv").write_text(
            res["candidates"].to_csv(index=False), encoding="utf-8")
        (v3.LOCAL_OUT / f"{sym}_context.csv").write_text(
            res["context"].to_csv(index=False), encoding="utf-8")
        (v3.LOCAL_OUT / f"{sym}_levels_full.csv").write_text(
            res["levels_full"].to_csv(index=False), encoding="utf-8")
        (v3.LOCAL_OUT / f"{sym}_future_path.csv").write_text(
            res["path"].to_csv(index=False), encoding="utf-8")
        (v3.LOCAL_OUT / f"{sym}_quantile_state.csv").write_text(
            res["quantile"].to_csv(index=False), encoding="utf-8")

        print(f"[v3-extend] {sym} DONE "
              f"{time.perf_counter()-t0:.0f}s "
              f"cand={res['candidates'].shape}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True)
    a = ap.parse_args()
    syms = [s.strip() for s in a.symbols.split(",") if s.strip()]
    sys.argv = [sys.argv[0]]
    run(syms)
    print("V3_EXTEND_DONE", flush=True)


if __name__ == "__main__":
    main()
