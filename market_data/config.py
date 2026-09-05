from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCE = "TqSdk"
SYMBOL = "KQ.m@SHFE.ag"
DISPLAY_NAME = "SHFE Silver Main Continuous"

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
