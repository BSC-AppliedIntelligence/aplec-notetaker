from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Settings


class RunStore:
    def __init__(self, settings: Settings):
        self.settings = settings

    def prepare_run_dir(self, meeting_id: int | str) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base_parent = self.settings.meetings_dir / str(meeting_id)
        candidate = base_parent / timestamp
        suffix = 1
        while True:
            try:
                candidate.mkdir(parents=True, exist_ok=False)
                return candidate
            except FileExistsError:
                candidate = base_parent / f"{timestamp}-{suffix}"
                suffix += 1

    def write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    def update_latest(self, meeting_id: int | str, payload: Any) -> None:
        latest_path = self.settings.meetings_dir / str(meeting_id) / "latest.json"
        self.write_json(latest_path, payload)

