"""Restore the established core code from the currently reviewed GitHub commit.

Use this only if the scaffold is placed into a completely new/empty directory and the
existing core files were not copied over. It intentionally downloads exact repository
files instead of asking AI to recreate canonical code.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
PINNED_COMMIT = "3ee6010f9a182038bca667e924081162b79b4c0c"
BASE = f"https://raw.githubusercontent.com/bao1872/future_dev/{PINNED_COMMIT}"

CORE_FILES = [
    "panji_indicators.py",
    "download_silver_main_tqsdk.py",
    "build_continuous.py",
    "visualize_smc_momentum_tqsdk.py",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="overwrite existing core files")
    args = parser.parse_args()

    for rel in CORE_FILES:
        target = ROOT / rel
        if target.exists() and not args.force:
            print(f"[keep] {rel}")
            continue
        url = f"{BASE}/{rel}"
        print(f"[download] {rel} <- {PINNED_COMMIT[:12]}")
        with urlopen(url, timeout=30) as resp:
            data = resp.read()
        target.write_bytes(data)

    print("Core restore complete.")
    print("Run: python scripts/check_project.py")


if __name__ == "__main__":
    main()
