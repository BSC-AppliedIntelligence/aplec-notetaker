"""The team roster: who may use fred-live, and with which Vexa identity.

Two sources merge into one allowlist:

* ``FRED_LIVE_WEB_USERS`` (env, ``email:vexa_user_id[:admin],...``) — the
  operator-managed bootstrap. Editing it requires restarting both apps, which
  is fine for the one admin entry it exists to guarantee.
* ``data/users.json`` — users added by ``python -m fred_live_worker.onboard``.
  Read on every merge call, so onboarding (and offboarding) takes effect on
  the next request with NO restarts: the web app resolves sessions and the
  worker re-checks capture tokens against this merged map per request.

On a conflicting email the env entry wins: it is the more privileged,
restart-gated source, and the tie only happens when someone re-onboards an
operator account.

``data/credentials/vexa_keys.json`` (0600) holds each user's user-scoped Vexa
API key, written at onboard time. The autojoin watcher launches each
calendar's bots with the calendar OWNER's key, which is what makes the
resulting meetings belong to that user in Vexa's DB — and therefore visible
to (only) them in the web app.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

USERS_FILE = "users.json"
VEXA_KEYS_FILE = "vexa_keys.json"  # under data/credentials/, chmod 0600


@dataclass(frozen=True)
class TeamUser:
    email: str
    vexa_user_id: int
    is_admin: bool = False


def parse_env_users(env_string: str) -> dict[str, TeamUser]:
    """``email:vexa_user_id[:admin],...`` -> map. Malformed entries are skipped."""
    out: dict[str, TeamUser] = {}
    for entry in (env_string or "").split(","):
        parts = [p.strip() for p in entry.split(":")]
        if len(parts) < 2 or not parts[0]:
            continue
        try:
            uid = int(parts[1])
        except ValueError:
            continue
        email = parts[0].lower()
        out[email] = TeamUser(email, uid, len(parts) > 2 and parts[2].lower() == "admin")
    return out


def _users_path(data_dir: Path) -> Path:
    return Path(data_dir) / USERS_FILE


def load_file_users(data_dir: Path) -> dict[str, TeamUser]:
    try:
        raw = json.loads(_users_path(data_dir).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        logger.warning("Unreadable %s; ignoring file users", _users_path(data_dir))
        return {}
    out: dict[str, TeamUser] = {}
    for email, entry in (raw.get("users") or {}).items():
        try:
            out[email.lower()] = TeamUser(
                email.lower(), int(entry["vexa_user_id"]), bool(entry.get("admin"))
            )
        except (KeyError, TypeError, ValueError):
            logger.warning("Skipping malformed users.json entry for %r", email)
    return out


def merged_users(env_string: str, data_dir: Path) -> dict[str, TeamUser]:
    merged = load_file_users(data_dir)
    merged.update(parse_env_users(env_string))  # env wins on conflict
    return merged


def _write_json_private(path: Path, payload: dict, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.chmod(tmp, mode)
    tmp.replace(path)


def add_file_user(data_dir: Path, email: str, vexa_user_id: int, admin: bool = False) -> None:
    path = _users_path(data_dir)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    users = raw.setdefault("users", {})
    users[email.lower()] = {"vexa_user_id": int(vexa_user_id), "admin": bool(admin)}
    _write_json_private(path, raw)


def remove_file_user(data_dir: Path, email: str) -> bool:
    path = _users_path(data_dir)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    removed = raw.get("users", {}).pop(email.lower(), None) is not None
    if removed:
        _write_json_private(path, raw)
    return removed


# --- per-user Vexa API keys --------------------------------------------------

def _keys_path(data_dir: Path) -> Path:
    return Path(data_dir) / "credentials" / VEXA_KEYS_FILE


def load_vexa_keys(data_dir: Path) -> dict[str, str]:
    try:
        raw = json.loads(_keys_path(data_dir).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        logger.warning("Unreadable %s; autojoin has no per-user keys", _keys_path(data_dir))
        return {}
    return {str(k).lower(): str(v) for k, v in raw.items() if v}


def store_vexa_key(data_dir: Path, email: str, api_key: str) -> None:
    path = _keys_path(data_dir)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    raw[email.lower()] = api_key
    _write_json_private(path, raw, mode=0o600)


def remove_vexa_key(data_dir: Path, email: str) -> bool:
    path = _keys_path(data_dir)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    removed = raw.pop(email.lower(), None) is not None
    if removed:
        _write_json_private(path, raw, mode=0o600)
    return removed
