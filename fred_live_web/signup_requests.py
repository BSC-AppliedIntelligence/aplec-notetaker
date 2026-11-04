"""Self-service access requests (the send-them-the-link onboarding flow).

Anyone who can reach the site (often a private network or VPN, itself a meaningful filter)
signs in with Google; an email that is not on the roster lands here as a
pending request instead of a flat denial. An admin approves it from
/admin/users, which runs the full server-side onboarding through the worker;
the person's next sign-in just works. Rejection deletes the request; signing
in again simply re-queues it (the population is VPN-gated, so queue spam is
not a real concern).

A pending entry grants NOTHING: no session is issued, and the email appears
nowhere but the admin queue.

State: data/signup_requests.json
  {email: {"name": str|None, "requested_at": iso, "last_seen_at": iso}}
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

_FILE = "signup_requests.json"


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


def add(data_dir: Path, email: str, name: str | None) -> None:
    """Insert, or refresh an existing request (keeps the original timestamp
    so the admin sees how long someone has been waiting)."""
    data = _load(data_dir)
    email = email.strip().lower()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry = data.get(email) or {"requested_at": now}
    entry["name"] = (name or entry.get("name") or "").strip() or None
    entry["last_seen_at"] = now
    data[email] = entry
    _save(data_dir, data)


def clear(data_dir: Path, email: str) -> bool:
    """Remove a request (approved and onboarded, or rejected)."""
    data = _load(data_dir)
    if data.pop(email.strip().lower(), None) is None:
        return False
    _save(data_dir, data)
    return True


def all_requests(data_dir: Path) -> dict:
    return _load(data_dir)
