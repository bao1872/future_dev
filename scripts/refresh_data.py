from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from market_data.pytdx_source import refresh_offline_market_data


if __name__ == "__main__":
    refresh_offline_market_data()
