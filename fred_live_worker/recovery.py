"""Startup recovery: reprocess runs orphaned by a worker restart.

A hook is acknowledged (202) and persisted as hook.json before processing starts;
if the worker dies mid-pipeline, the run dir is left with hook.json but neither
result.json nor error.json — and Vexa will not redeliver (it saw the 202).
On startup we sweep for such orphans and re-enqueue them.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

from .schemas import MeetingCompletedPayload

logger = logging.getLogger(__name__)

_TERMINAL_FILES = ("result.json", "error.json")


def find_orphaned_runs(
    meetings_dir: Path,
    min_age_seconds: float = 60.0,
    max_age_seconds: float = 3 * 24 * 3600.0,
) -> list[Path]:
    """Run dirs containing hook.json but no terminal file, aged between the bounds.
    The max-age bound keeps a sweep from resurrecting ancient orphans (pre-migration
    debris, months-old tests) into expensive GPU reprocessing. Pure function;
    sorted oldest-first so recovery preserves arrival order."""
    if not meetings_dir.is_dir():
        return []
    now = time.time()
    orphans: list[Path] = []
    for meeting_dir in meetings_dir.iterdir():
        if not meeting_dir.is_dir():
            continue
        for run_dir in meeting_dir.iterdir():
            if not run_dir.is_dir():
                continue
            hook = run_dir / "hook.json"
            if not hook.exists():
                continue
            if any((run_dir / name).exists() for name in _TERMINAL_FILES):
                continue
            age = now - hook.stat().st_mtime
            if age < min_age_seconds or age > max_age_seconds:
                continue
            orphans.append(run_dir)
    return sorted(orphans, key=lambda p: p.name)


async def recover_orphans(processor, settings, max_runs: int = 10) -> int:
    """Reprocess orphaned runs serially. Returns the number recovered.
    Each orphan gets an error.json breadcrumb in its OLD dir pointing at recovery,
    so repeated sweeps (or the next restart) don't pick it up again."""
    orphans = find_orphaned_runs(settings.meetings_dir)
    if not orphans:
        return 0
    if len(orphans) > max_runs:
        logger.warning(
            "Recovery sweep found %d orphaned runs; capping at %d", len(orphans), max_runs
        )
        orphans = orphans[:max_runs]

    recovered = 0
    for run_dir in orphans:
        try:
            payload = MeetingCompletedPayload.model_validate(
                json.loads((run_dir / "hook.json").read_text(encoding="utf-8"))
            )
        except Exception:
            logger.exception("Recovery: unreadable hook.json in %s; skipping", run_dir)
            continue

        (run_dir / "error.json").write_text(
            json.dumps(
                {
                    "error_type": "OrphanedRun",
                    "message": "Worker restarted before this run finished; "
                    "reprocessed by the startup recovery sweep in a new run dir.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Recovery: reprocessing meeting %s from %s", payload.meeting.id, run_dir)
        try:
            await processor.process(payload)
            recovered += 1
        except Exception:
            logger.exception("Recovery: reprocessing failed for %s", run_dir)
    return recovered


def schedule_recovery(processor, settings) -> None:
    """Fire-and-forget recovery task from FastAPI startup."""

    async def _run() -> None:
        try:
            count = await recover_orphans(processor, settings)
            if count:
                logger.info("Recovery sweep reprocessed %d orphaned run(s)", count)
        except Exception:
            logger.exception("Recovery sweep failed")

    asyncio.get_event_loop().create_task(_run())
