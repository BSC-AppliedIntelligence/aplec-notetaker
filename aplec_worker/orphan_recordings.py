"""Orphaned-recording sweep: audio on disk that no run ever consumed.

recovery.py closes the window AFTER a hook is persisted (a run dir holding
hook.json but no result). This closes the window BEFORE it: audio that exists
on disk for which no hook was ever written at all, so NOTHING in the system
knows the meeting happened. It never appears in the history, not even as a
failure, and no sweep looks for it.

Two ways that happens:
  * an extension capture finished, its processing task was queued in-process
    (FastAPI BackgroundTasks), and the worker died before the task ran — the
    202 was already sent, the recording is complete, and the intent to process
    it existed only in memory;
  * Vexa tried to deliver meeting.completed while the worker was down past its
    retry window — the bot's recording is on disk, the payload is gone forever.

Captures carry their own metadata (data/captures/<sid>/meta.json), so an
orphan with real audio is RECOVERABLE: finish_payload() rebuilds exactly what
the finish endpoint would have produced, speaker events included. Vexa
recordings are not — meeting id, participants and speaker events lived only in
the undelivered payload — so those are reported for the operator instead of
being silently ignored.

Empty recordings (a capture that never got mic audio, a bot that failed to
join) are counted but not reported as lost meetings: there is nothing in them.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .retention import _tree_newest_mtime  # same "how old is this tree" question

logger = logging.getLogger(__name__)

# Below this a recording holds no meeting: 32KB of opus is ~2 seconds. Failed
# captures land at exactly 0 bytes, but a hair above zero is just as empty.
MIN_RECOVERABLE_BYTES = 32 * 1024
SWEEP_INTERVAL_SECONDS = 6 * 3600
REPORT_NAME = "orphaned_recordings.json"
ATTEMPTS_NAME = "orphan_recoveries.json"


def processed_meeting_ids(meetings_dir: Path) -> set[str]:
    """Meeting ids that have at least one persisted run. A FAILED run counts:
    the meeting is visible in the history with an error card, so it is not
    lost. This sweep is about silence, not about failure."""
    if not meetings_dir.is_dir():
        return set()
    seen: set[str] = set()
    for meeting_dir in meetings_dir.iterdir():
        if not meeting_dir.is_dir():
            continue
        for run_dir in meeting_dir.iterdir():
            if run_dir.is_dir() and (run_dir / "hook.json").exists():
                seen.add(meeting_dir.name)
                break
    return seen


def referenced_session_uids(meetings_dir: Path) -> set[str]:
    """Vexa session uids named by any saved hook's storage_path.

    Bot recordings live at recordings/<user>/<storage_id>/<session_uid>/<media>/
    and the uid is a per-session UUID, so it identifies the tree even though
    the run dir is named after the meeting id (an unrelated number)."""
    uids: set[str] = set()
    if not meetings_dir.is_dir():
        return uids
    for hook in meetings_dir.glob("*/*/hook.json"):
        try:
            payload = json.loads(hook.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        meeting = (payload.get("data") or {}).get("meeting") or payload.get("meeting") or {}
        for recording in ((meeting.get("data") or {}).get("recordings") or []):
            for media in recording.get("media_files") or []:
                path = str(media.get("storage_path") or "")
                parts = [p for p in path.replace("\\", "/").split("/") if p]
                # .../<storage_id>/<session_uid>/<media_type>/master.webm
                if len(parts) >= 3:
                    uids.add(parts[-3])
    return uids


def find_orphaned_captures(settings, min_age_minutes: float = 120.0) -> list[dict]:
    """Extension captures whose audio exists but which produced no run."""
    root = settings.recordings_dir / "captures"
    if not root.is_dir():
        return []
    processed = processed_meeting_ids(settings.meetings_dir)
    cutoff = time.time() - min_age_minutes * 60
    orphans: list[dict] = []
    for session in sorted(root.iterdir()):
        if not session.is_dir() or session.name in processed:
            continue
        try:
            newest = _tree_newest_mtime(session)
            size = sum(f.stat().st_size for f in session.rglob("*") if f.is_file())
        except OSError:
            continue
        if newest > cutoff:  # still being written, or processing right now
            continue
        orphans.append(
            {
                "kind": "capture",
                "id": session.name,
                "path": str(session),
                "size_bytes": size,
                "age_hours": round((time.time() - newest) / 3600, 1),
                "recoverable": size >= MIN_RECOVERABLE_BYTES,
            }
        )
    return orphans


def find_orphaned_vexa_recordings(settings, min_age_minutes: float = 120.0) -> list[dict]:
    """Bot recording trees no saved hook refers to (undelivered webhook)."""
    root = settings.recordings_dir / "recordings"
    if not root.is_dir():
        return []
    referenced = referenced_session_uids(settings.meetings_dir)
    cutoff = time.time() - min_age_minutes * 60
    orphans: list[dict] = []
    for user_dir in sorted(root.iterdir()):
        if not user_dir.is_dir():
            continue
        for storage in sorted(user_dir.iterdir()):
            if not storage.is_dir():
                continue
            for session in sorted(storage.iterdir()):
                if not session.is_dir() or session.name in referenced:
                    continue
                try:
                    newest = _tree_newest_mtime(session)
                    size = sum(f.stat().st_size for f in session.rglob("*") if f.is_file())
                except OSError:
                    continue
                if newest > cutoff:
                    continue
                orphans.append(
                    {
                        "kind": "vexa",
                        "id": session.name,
                        "path": str(session),
                        "size_bytes": size,
                        "age_hours": round((time.time() - newest) / 3600, 1),
                        # The payload that named this meeting is gone; only an
                        # operator can decide what it was.
                        "recoverable": False,
                    }
                )
    return orphans


def find_orphans(settings, min_age_minutes: float = 120.0) -> list[dict]:
    return find_orphaned_captures(settings, min_age_minutes) + find_orphaned_vexa_recordings(
        settings, min_age_minutes
    )


def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


async def sweep_orphans(
    processor,
    settings,
    min_age_minutes: float = 120.0,
    max_recover_age_days: float = 3.0,
    max_recoveries: int = 3,
) -> dict:
    """One pass: find orphans, recover the recoverable ones, write the report.

    Recovery is attempted ONCE per session (recorded in orphan_recoveries.json)
    — a capture that fails to process would otherwise re-burn a GPU pass every
    sweep, forever. Success creates the run dir, which removes it from the
    orphan set anyway; the ledger is there for the failures."""
    from .capture import CaptureStore
    from .schemas import MeetingCompletedPayload

    orphans = find_orphans(settings, min_age_minutes)
    report_path = settings.data_dir / REPORT_NAME
    attempts_path = settings.data_dir / ATTEMPTS_NAME
    previous = {o.get("id") for o in _load_json(report_path, {}).get("orphans", [])}
    attempts: dict = _load_json(attempts_path, {})

    store = CaptureStore(settings)
    recovered = 0
    for orphan in orphans:
        if orphan["kind"] != "capture" or not orphan["recoverable"]:
            continue
        if orphan["id"] in attempts:
            orphan["status"] = attempts[orphan["id"]].get("status", "attempted")
            continue
        if orphan["age_hours"] > max_recover_age_days * 24:
            orphan["status"] = "too-old-to-recover"
            continue
        if recovered >= max_recoveries:
            orphan["status"] = "queued-for-next-sweep"
            continue
        logger.warning(
            "Orphan sweep: capture %s (%.1f MB, %.1fh old) has audio but no run; recovering",
            orphan["id"], orphan["size_bytes"] / 1e6, orphan["age_hours"],
        )
        status = "recovered"
        try:
            payload = MeetingCompletedPayload.model_validate(store.finish_payload(orphan["id"]))
            await processor.process(payload)
            recovered += 1
        except Exception:
            logger.exception("Orphan sweep: recovery failed for capture %s", orphan["id"])
            status = "recovery-failed"
        orphan["status"] = status
        attempts[orphan["id"]] = {"status": status, "attempted_at": int(time.time())}
        _write_json(attempts_path, attempts)

    lost = [o for o in orphans if o["recoverable"] and o.get("status") != "recovered"]
    empty = [o for o in orphans if not o["recoverable"]]
    for orphan in lost:
        if orphan["id"] not in previous:
            logger.warning(
                "Orphan sweep: %s recording %s (%.1f MB) has no run and cannot be "
                "auto-recovered; inspect %s",
                orphan["kind"], orphan["id"], orphan["size_bytes"] / 1e6, orphan["path"],
            )
    _write_json(
        report_path,
        {
            "generated_at": int(time.time()),
            "orphans": orphans,
            "summary": {
                "total": len(orphans),
                "recovered_this_sweep": recovered,
                "unrecovered_with_audio": len(lost),
                "empty": len(empty),
            },
        },
    )
    if orphans:
        logger.info(
            "Orphan sweep: %d orphaned recording(s) — %d recovered, %d still lost, %d empty",
            len(orphans), recovered, len(lost), len(empty),
        )
    return {"orphans": orphans, "recovered": recovered, "lost": len(lost), "empty": len(empty)}


def schedule_orphan_sweep(processor, settings) -> None:
    """Periodic sweep as a worker background task."""
    import asyncio

    if not settings.orphan_sweep_enabled:
        logger.info("Orphan-recording sweep disabled (APLEC_ORPHAN_SWEEP_ENABLED=false)")
        return

    async def _loop() -> None:
        # After the startup recovery sweep: a run it revives stops being an
        # orphan, and we must not race it into a duplicate GPU pass.
        await asyncio.sleep(300)
        while True:
            try:
                await sweep_orphans(
                    processor, settings, min_age_minutes=settings.orphan_min_age_minutes
                )
            except Exception:
                logger.exception("Orphan-recording sweep failed; retrying later")
            await asyncio.sleep(SWEEP_INTERVAL_SECONDS)

    asyncio.get_event_loop().create_task(_loop())


def _main() -> None:
    """Read-only report: python -m aplec_worker.orphan_recordings"""
    from .config import get_settings

    settings = get_settings()
    orphans = find_orphans(settings, min_age_minutes=0.0)
    if not orphans:
        print("No orphaned recordings: every recording on disk has a run.")
        return
    for orphan in orphans:
        flag = "RECOVERABLE" if orphan["recoverable"] else "empty      "
        print(
            f"{flag}  {orphan['kind']:<8} {orphan['id']:<40} "
            f"{orphan['size_bytes'] / 1e6:8.1f} MB  {orphan['age_hours']:7.1f}h  {orphan['path']}"
        )
    print(f"\n{len(orphans)} orphaned recording(s); "
          f"{sum(1 for o in orphans if o['recoverable'])} hold audio.")


if __name__ == "__main__":
    _main()
