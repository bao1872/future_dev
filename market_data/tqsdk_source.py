from __future__ import annotations

from pathlib import Path


def refresh_offline_market_data() -> None:
    """Refresh current offline market files using the established downloader.

    Governance boundary: this module is the market-data source layer. Strategy modules
    must not call this function or import TqSdk directly.
    """
    root = Path(__file__).resolve().parents[1]
    core = root / "download_silver_main_tqsdk.py"
    if not core.is_file():
        raise RuntimeError(
            "Missing established downloader: download_silver_main_tqsdk.py. "
            "Restore the current repository core file; do not replace it with a new implementation."
        )

    from download_silver_main_tqsdk import main

    main()
