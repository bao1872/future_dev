from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_CORE = [
    "panji_indicators.py",
    "download_silver_main_tqsdk.py",
    "build_continuous.py",
    "visualize_smc_momentum_tqsdk.py",
]

REQUIRED_SCAFFOLD = [
    "AGENTS.md",
    "app.py",
    "market_data/offline_store.py",
    "market_data/tqsdk_source.py",
    "market_data/validation.py",
    "strategies/registry.py",
]

missing = []
for rel in REQUIRED_CORE + REQUIRED_SCAFFOLD:
    if not (ROOT / rel).is_file():
        missing.append(rel)

if missing:
    print("PROJECT CHECK: FAIL")
    for rel in missing:
        print(f"  missing: {rel}")
    print("\nDo not replace canonical/core files with placeholders. Restore them from the current repository.")
    sys.exit(1)

print("PROJECT CHECK: PASS")
print("Core existing assets and research scaffold are present.")

adjusted = ROOT / "silver_main_data" / "adjusted"
if adjusted.exists():
    print("WARNING: adjusted/ exists. Do not assume it is current/valid without rebuilding and validation.")
