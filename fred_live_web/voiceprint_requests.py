"""Voiceprint enrollment invitations (the leader-invites, member-consents flow).

The team leader (admin) requests enrollment; the member sees the request in
their Settings, reads the consent notice, and records/uploads THEIR OWN audio
(or declines). The leader never touches anyone else's voice: that inversion is
what keeps the feature consent-based under GDPR Art. 9 (explicit consent),
with the no-detriment property (declining changes nothing about meetings).

State: data/voiceprint_requests.json
  {email: {"requested_by": email, "requested_at": iso, "status": "pending"|"declined"}}
Enrollment success clears the entry (the print itself, held by the worker,
carries the consent metadata from then on).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

# Bump when the consent notice text changes materially; stored inside each
# print's metadata so we know which notice the person saw.
NOTICE_VERSION = "v2-2026-10"

_FILE = "voiceprint_requests.json"


def _path(data_dir: Path) -> Path:
    return Path(data_dir) / _FILE


def _load(data_dir: Path) -> dict:
    try:
        data = json.loads(_path(data_dir).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save(data_dir: Path, data: dict) -> None:
    path = _path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def request_enrollment(data_dir: Path, member_email: str, requested_by: str) -> None:
    data = _load(data_dir)
    data[member_email.strip().lower()] = {
        "requested_by": requested_by.strip().lower(),
        "requested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "pending",
    }
    _save(data_dir, data)


def decline(data_dir: Path, member_email: str) -> bool:
    data = _load(data_dir)
    entry = data.get(member_email.strip().lower())
    if not entry:
        return False
    entry["status"] = "declined"
    entry["declined_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _save(data_dir, data)
    return True


def clear(data_dir: Path, member_email: str) -> bool:
    """Remove the entry entirely (enrollment completed, or leader cancels)."""
    data = _load(data_dir)
    if data.pop(member_email.strip().lower(), None) is None:
        return False
    _save(data_dir, data)
    return True


def get(data_dir: Path, member_email: str) -> dict | None:
    return _load(data_dir).get(member_email.strip().lower())


def pending_for(data_dir: Path, member_email: str) -> bool:
    entry = get(data_dir, member_email)
    return bool(entry and entry.get("status") == "pending")


def all_requests(data_dir: Path) -> dict:
    return _load(data_dir)
