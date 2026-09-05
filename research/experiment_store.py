from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "research" / "results"


def current_git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def save_experiment(
    *,
    strategy: str,
    params: dict[str, Any],
    data_start: str,
    data_end: str,
    result: dict[str, Any],
    note: str = "",
) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now().astimezone()
    experiment_id = now.strftime("EXP-%Y%m%d-%H%M%S")
    payload = {
        "experiment_id": experiment_id,
        "created_at": now.isoformat(),
        "strategy": strategy,
        "params": params,
        "data_start": data_start,
        "data_end": data_end,
        "git_sha": current_git_sha(),
        "result": result,
        "note": note,
    }
    path = RESULT_DIR / f"{experiment_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def list_experiments() -> list[dict[str, Any]]:
    if not RESULT_DIR.is_dir():
        return []
    rows = []
    for path in sorted(RESULT_DIR.glob("*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["_path"] = str(path)
            rows.append(payload)
        except Exception:
            continue
    return rows
