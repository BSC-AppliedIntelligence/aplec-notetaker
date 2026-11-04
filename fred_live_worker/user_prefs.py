"""Per-user preferences, shared between the web app (writes) and the worker
(reads). No database: data/user_prefs.json keyed by Vexa user id — the one
identity both sides know (the web maps emails to it, meetings carry it).

Preferences: the summary output language, chosen at onboarding or in
Settings ('same-as-transcript', the default, resolves to the meeting's
dominant language at summarize time); the autojoin policy; automatic
sharing with the people recognized by voice (on unless turned off); and the
site's interface language (web only).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PREFS_FILENAME = "user_prefs.json"

# value -> UI label. Keep insertion order: it is the dropdown order.
SUMMARY_LANGUAGES: dict[str, str] = {
    "same-as-transcript": "Same as the meeting (auto)",
    "en": "English",
    "es": "Spanish",
    "pt-BR": "Brazilian Portuguese",
    "it": "Italian",
    "fr": "French",
    "de": "German",
    "ca": "Catalan",
    "pl": "Polish",
    "ru": "Russian",
}


def _path(data_dir: Path) -> Path:
    return data_dir / PREFS_FILENAME


def _load(data_dir: Path) -> dict:
    try:
        data = json.loads(_path(data_dir).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


# value -> UI label. "accepted" is the default: Fred only covers meetings the
# user organized or explicitly accepted — invited-but-ignored team meetings
# don't get an uninvited notetaker.
AUTOJOIN_POLICIES: dict[str, str] = {
    "accepted": "Meetings I organized or accepted",
    "all": "Every meeting with a Meet link",
    "off": "Never join automatically",
}
DEFAULT_AUTOJOIN_POLICY = "accepted"


def get_autojoin_policy(data_dir: Path, email: str | None) -> str:
    """Autojoin policy for a calendar owner's email. Prefs entries record the
    user's email (written by the web app), which is the identity the watcher
    has — calendar token files are named after it."""
    if email:
        for entry in _load(data_dir).values():
            if isinstance(entry, dict) and entry.get("email", "").lower() == email.lower():
                value = entry.get("autojoin_policy")
                if value in AUTOJOIN_POLICIES:
                    return value
                break
    return DEFAULT_AUTOJOIN_POLICY


def set_autojoin_policy(data_dir: Path, vexa_user_id, email: str, value: str) -> None:
    if value not in AUTOJOIN_POLICIES:
        raise ValueError(f"Unsupported autojoin policy: {value!r}")
    prefs = _load(data_dir)
    user = prefs.setdefault(str(vexa_user_id), {})
    user["autojoin_policy"] = value
    user["email"] = email
    path = _path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prefs, ensure_ascii=False, indent=1), encoding="utf-8")


def get_auto_share(data_dir: Path, vexa_user_id) -> bool:
    """Share my recordings with the people Fred recognizes by voice in them.
    On unless the user turned it off."""
    if vexa_user_id is None:
        return True
    return (_load(data_dir).get(str(vexa_user_id)) or {}).get("auto_share") is not False


def set_auto_share(data_dir: Path, vexa_user_id, value: bool, email: str | None = None) -> None:
    prefs = _load(data_dir)
    user = prefs.setdefault(str(vexa_user_id), {})
    user["auto_share"] = bool(value)
    if email:
        user["email"] = email
    path = _path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prefs, ensure_ascii=False, indent=1), encoding="utf-8")


# The site's own interface language (fred_live_web/i18n.py), chosen in Settings.
# The worker never reads it: notes follow the summary language above.
UI_LANGUAGES = ("en", "es", "ca")


def get_ui_language(data_dir: Path, vexa_user_id) -> str | None:
    """The interface language the user chose, or None (the site then uses English)."""
    if vexa_user_id is None:
        return None
    value = (_load(data_dir).get(str(vexa_user_id)) or {}).get("ui_language")
    return value if value in UI_LANGUAGES else None


def set_ui_language(data_dir: Path, vexa_user_id, value: str, email: str | None = None) -> None:
    if value not in UI_LANGUAGES:
        raise ValueError(f"Unsupported interface language: {value!r}")
    prefs = _load(data_dir)
    user = prefs.setdefault(str(vexa_user_id), {})
    user["ui_language"] = value
    if email:
        user["email"] = email
    path = _path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prefs, ensure_ascii=False, indent=1), encoding="utf-8")


def get_summary_language(data_dir: Path, vexa_user_id) -> str | None:
    """The user's chosen output language, or None for default behavior."""
    if vexa_user_id is None:
        return None
    value = (_load(data_dir).get(str(vexa_user_id)) or {}).get("summary_language")
    if value in (None, "", "same-as-transcript"):
        return None
    if value not in SUMMARY_LANGUAGES:
        logger.warning("Ignoring unknown summary_language %r for user %s", value, vexa_user_id)
        return None
    return value


def set_summary_language(data_dir: Path, vexa_user_id, value: str, email: str | None = None) -> None:
    if value not in SUMMARY_LANGUAGES:
        raise ValueError(f"Unsupported summary language: {value!r}")
    prefs = _load(data_dir)
    user = prefs.setdefault(str(vexa_user_id), {})
    user["summary_language"] = value
    if email:
        user["email"] = email
    path = _path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prefs, ensure_ascii=False, indent=1), encoding="utf-8")
