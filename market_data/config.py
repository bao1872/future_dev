from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCE = "PyTDX"
SYMBOL = "AGL8"
DISPLAY_NAME = "SHFE Silver Main Continuous (PyTDX L8)"

# Bar period in seconds.
# Layout is replaced by the PyTDX 5m source bar in the rebaseline
# commit; do not change it here or the offline store breaks before
# the new dataset exists.
TIMEFRAMES = {
    "15m": 15 * 60,
    "1h": 60 * 60,
    "4h": 4 * 60 * 60,
}

DATA_DIR = ROOT / "silver_main_data"
CURRENT_FILES = {
    tf: DATA_DIR / f"silver_main_{tf}.csv"
    for tf in TIMEFRAMES
}
