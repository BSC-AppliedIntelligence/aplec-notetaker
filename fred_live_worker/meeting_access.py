"""Per-meeting read access: ``data/meetings/<id>/access.json``.

Grants are per-meeting and keyed by EMAIL (the roster's identity), stored next
to the run dirs and never inside ``result.json``, which the pipeline REWRITES
on every reprocess (grants stored there would silently vanish).

Two writers share this file, so it lives in the worker package, which the web
image also carries:

* the web app: an owner or admin shares a meeting with a person or a team
  snapshot (fred_live_web/sharing.py), or removes someone;
* the worker: automatic sharing with the registered users recognized BY VOICE
  in the meeting (``via: "auto:voice"``; pipeline ``_auto_share``).

A removal is remembered (``revoked``): automatic sharing never re-adds someone
the owner took out, not even when the meeting is reprocessed. Sharing that
person again by hand clears it.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

_MEETING_ID_RE = re.compile(r"^[\w-]+$")
AUTO_VOICE = "auto:voice"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def _access_path(data_dir: Path, meeting_id: str) -> Path | None:
    if not _MEETING_ID_RE.fullmatch(str(meeting_id)):
        return None
    return Path(data_dir) / "meetings" / str(meeting_id) / "access.json"


def grants_for(data_dir: Path, meeting_id: str) -> dict:
    """{email: {"granted_by", "granted_at", "via"}}. Empty dict when none."""
    path = _access_path(data_dir, meeting_id)
    if path is None:
        return {}
    return _load_json(path).get("grants", {})


def granted_emails(data_dir: Path, meeting_id: str) -> set[str]:
    return set(grants_for(data_dir, meeting_id))


def revoked_emails(data_dir: Path, meeting_id: str) -> set[str]:
    path = _access_path(data_dir, meeting_id)
    if path is None:
        return set()
    return set(_load_json(path).get("revoked", []))


def grant(
    data_dir: Path,
    meeting_id: str,
    emails: list[str],
    granted_by: str,
    via: str = "direct",
) -> int:
    """Add viewer grants (idempotent per email; existing grants keep their
    original metadata). The meeting dir must already exist: you can only
    share a meeting that has at least a run dir. A manual grant also clears
    an earlier removal, so automatic sharing may add the person again later.
    Returns grants added."""
    path = _access_path(data_dir, meeting_id)
    if path is None or not path.parent.is_dir():
        return 0
    data = _load_json(path)
    grants = data.setdefault("grants", {})
    revoked = set(data.get("revoked", []))
    added = 0
    changed = False
    for email in emails:
        email = email.strip().lower()
        if not email:
            continue
        if via != AUTO_VOICE and email in revoked:
            revoked.discard(email)
            changed = True
        if email in grants:
            continue
        grants[email] = {"granted_by": granted_by.lower(), "granted_at": _now(), "via": via}
        added += 1
    if added or changed:
        data["revoked"] = sorted(revoked)
        if not data["revoked"]:
            data.pop("revoked")
        _save_json(path, data)
    return added


def auto_grant(data_dir: Path, meeting_id: str, emails: list[str], granted_by: str) -> list[str]:
    """Automatic sharing (recognized voices). Skips people already granted and
    people the owner removed from this meeting. Returns the emails added."""
    skip = revoked_emails(data_dir, meeting_id) | granted_emails(data_dir, meeting_id)
    fresh = sorted({e.strip().lower() for e in emails if e.strip()} - skip)
    if not fresh or not grant(data_dir, meeting_id, fresh, granted_by, via=AUTO_VOICE):
        return []  # nothing new, or no such meeting dir / unsafe id
    return fresh


def revoke(data_dir: Path, meeting_id: str, email: str) -> bool:
    """Remove a viewer and remember it, so automatic sharing never re-adds them."""
    path = _access_path(data_dir, meeting_id)
    if path is None:
        return False
    email = email.strip().lower()
    data = _load_json(path)
    removed = data.get("grants", {}).pop(email, None) is not None
    if removed:
        data["revoked"] = sorted(set(data.get("revoked", [])) | {email})
        _save_json(path, data)
    return removed


def meetings_shared_with(data_dir: Path, email: str) -> set[str]:
    """Meeting ids this email may view as a shared VIEWER. One small file per
    shared meeting; only meetings someone bothered to share have the file at
    all, so this stays cheap."""
    email = email.strip().lower()
    out: set[str] = set()
    meetings_dir = Path(data_dir) / "meetings"
    if not meetings_dir.is_dir():
        return out
    for path in meetings_dir.glob("*/access.json"):
        if email in _load_json(path).get("grants", {}):
            out.add(path.parent.name)
    return out
