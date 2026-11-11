"""Recording retention: delete meeting AUDIO older than N days.

Transcripts, summaries and run metadata under data/meetings are the product
and are never touched — only the heavy recording trees under
APLEC_RECORDINGS_DIR expire. Vexa's meeting-api writes recordings as uid
999 inside the container, so the host worker cannot unlink them directly;
deletion is executed through `docker exec` in the meeting-api container (the
same trick manual cleanups already needed). A missing container or docker
binary logs a warning and skips — never crashes the worker.
"""
from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

CONTAINER_ROOT = "/data/recordings"
SWEEP_INTERVAL_SECONDS = 24 * 3600


def _tree_newest_mtime(path: Path) -> float:
    """Newest mtime in a tree — a session is only expired when NOTHING in it
    is fresh (reprocessing never touches recordings, so this is meeting age)."""
    newest = path.stat().st_mtime
    for child in path.rglob("*"):
        try:
            newest = max(newest, child.stat().st_mtime)
        except OSError:
            continue
    return newest


def find_expired(recordings_dir: Path, cutoff_epoch: float) -> list[Path]:
    """Expired recording trees: recordings/<user>/<storage>/<session> dirs and
    stray legacy .webm files whose content is entirely older than the cutoff."""
    root = recordings_dir / "recordings"
    expired: list[Path] = []
    if not root.is_dir():
        return expired
    for user_dir in root.iterdir():
        if not user_dir.is_dir():
            continue
        for storage in user_dir.iterdir():
            try:
                if storage.is_file():  # legacy flat <storage_id>/<session>.webm layout
                    if storage.stat().st_mtime < cutoff_epoch:
                        expired.append(storage)
                    continue
                for session in storage.iterdir():
                    if _tree_newest_mtime(session) < cutoff_epoch:
                        expired.append(session)
            except OSError:
                logger.warning("Retention: cannot stat %s; skipping", storage, exc_info=True)
    return expired


def to_container_path(host_path: Path, recordings_dir: Path) -> str:
    """Map a host recording path onto the meeting-api container mount."""
    rel = host_path.resolve().relative_to(recordings_dir.resolve())
    return f"{CONTAINER_ROOT}/{rel.as_posix()}"


def find_expired_captures(recordings_dir: Path, cutoff_epoch: float) -> list[Path]:
    """Expired extension-capture audio trees (recordings/captures/<sid>/).
    Same audio-only promise as bot recordings; the session metadata under
    data/captures (events, diagnostics) is small and stays, like transcripts."""
    root = recordings_dir / "captures"
    expired: list[Path] = []
    if not root.is_dir():
        return expired
    for session in root.iterdir():
        try:
            if session.is_dir() and _tree_newest_mtime(session) < cutoff_epoch:
                expired.append(session)
        except OSError:
            logger.warning("Retention: cannot stat %s; skipping", session, exc_info=True)
    return expired


def sweep_captures(settings) -> int:
    """Delete expired extension-capture audio. These trees are written by the
    worker itself (host uid), so plain rmtree works — no docker exec needed."""
    import shutil

    if settings.retention_days <= 0:
        return 0
    cutoff = time.time() - settings.retention_days * 24 * 3600
    deleted = 0
    for session in find_expired_captures(settings.recordings_dir, cutoff):
        resolved = session.resolve()
        if not resolved.is_relative_to(settings.recordings_dir.resolve()):
            logger.error("Retention: refusing out-of-root path %s", session)
            continue
        try:
            shutil.rmtree(resolved)
            deleted += 1
        except OSError:
            logger.warning("Retention: failed to delete capture %s", session, exc_info=True)
    if deleted:
        logger.info(
            "Retention: deleted %d extension-capture audio tree(s) older than %d days",
            deleted, settings.retention_days,
        )
    return deleted


def sweep(settings) -> int:
    """One retention pass. Returns the number of trees deleted."""
    if settings.retention_days <= 0:
        return 0
    cutoff = time.time() - settings.retention_days * 24 * 3600
    expired = find_expired(settings.recordings_dir, cutoff)
    if not expired:
        logger.info("Retention: nothing older than %d days", settings.retention_days)
        return 0

    container_paths = []
    for path in expired:
        # Belt and suspenders: never let anything outside the recordings tree
        # into the rm command.
        resolved = path.resolve()
        if not resolved.is_relative_to(settings.recordings_dir.resolve()):
            logger.error("Retention: refusing out-of-root path %s", path)
            continue
        container_paths.append(to_container_path(path, settings.recordings_dir))
    if not container_paths:
        return 0

    # Pass paths as argv, not interpolated into an `sh -c` string: a path
    # containing a quote would break the shell command, and argv removes any
    # possibility of shell metacharacter interpretation. `rm -rf --` stops
    # option parsing so a path can never be read as a flag. The empty-dir
    # prune is a second exec (find needs no per-path args).
    try:
        proc = subprocess.run(
            ["docker", "exec", "-u", "0", settings.retention_container,
             "rm", "-rf", "--", *container_paths],
            capture_output=True, text=True, timeout=120,
        )
        subprocess.run(
            ["docker", "exec", "-u", "0", settings.retention_container,
             "find", f"{CONTAINER_ROOT}/recordings", "-mindepth", "1",
             "-type", "d", "-empty", "-delete"],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("Retention: docker exec unavailable (%s); skipping sweep", exc)
        return 0
    if proc.returncode != 0:
        logger.warning(
            "Retention: container deletion failed (rc=%s): %s",
            proc.returncode, (proc.stderr or "")[:300],
        )
        return 0
    logger.info(
        "Retention: deleted %d recording tree(s) older than %d days: %s",
        len(container_paths), settings.retention_days,
        ", ".join(p.name for p in expired[:10]) + (" …" if len(expired) > 10 else ""),
    )
    return len(container_paths)


def schedule_retention(settings) -> None:
    """Daily sweep as a worker background task (first pass shortly after boot)."""
    import asyncio

    if settings.retention_days <= 0:
        logger.info("Retention disabled (APLEC_RETENTION_DAYS=0)")
        return

    async def _loop() -> None:
        await asyncio.sleep(120)  # let the boot recovery sweep go first
        while True:
            try:
                await asyncio.to_thread(sweep, settings)
            except Exception:
                logger.exception("Retention sweep failed; retrying tomorrow")
            try:
                await asyncio.to_thread(sweep_captures, settings)
            except Exception:
                logger.exception("Capture retention sweep failed; retrying tomorrow")
            await asyncio.sleep(SWEEP_INTERVAL_SECONDS)

    asyncio.get_event_loop().create_task(_loop())
