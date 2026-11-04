"""Meeting sharing: Teams (admin-managed groups) + per-meeting access grants.

Model:

* A meeting is visible to its OWNER (meeting.user_id) and admins, as before.
  Sharing adds VIEWERS: read access to the card, summary, transcript and
  ask-chat — never Reprocess, speaker renaming, or re-sharing.
* Grants are per-meeting and keyed by EMAIL (the roster's identity), stored in
  ``data/meetings/<id>/access.json`` — next to the run dirs, never inside
  ``result.json``, which the pipeline REWRITES on every reprocess (grants
  stored there would silently vanish).
* Sharing with a Team is SNAPSHOT semantics: the team's member list is
  resolved at share time and stored as individual grants (tagged with the
  team for display). Someone joining the team later does NOT gain access to
  previously shared meetings; removing a grant is always per-person.
  Consequence: ``teams.json`` is only ever read at share time.
* Teams live in ``data/teams.json``, managed by admins on /admin/users.
* AUTOMATIC sharing: the worker grants read access
  to the registered users it recognized BY VOICE in the meeting (``via:
  "auto:voice"``), unless the owner turned it off in Settings. Removing such a
  person is remembered: automatic sharing never adds them back.
"""
from __future__ import annotations

import re
from pathlib import Path

# Per-meeting grants are written by the worker too (automatic sharing with the
# people recognized by voice), so their storage lives in the worker package;
# re-exported here for the web routes.
from fred_live_worker.meeting_access import (  # noqa: F401
    AUTO_VOICE,
    _load_json,
    _save_json,
    grant,
    granted_emails,
    grants_for,
    meetings_shared_with,
    revoke,
)

TEAMS_FILE = "teams.json"


# --- teams -------------------------------------------------------------------

def _teams_path(data_dir: Path) -> Path:
    return Path(data_dir) / TEAMS_FILE


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return s[:40]


def load_teams(data_dir: Path) -> dict:
    """{slug: {"name": str, "members": [emails]}}"""
    return _load_json(_teams_path(data_dir)).get("teams", {})


def save_team(data_dir: Path, name: str, members: list[str]) -> str | None:
    """Create or replace a team (idempotent by slug). Returns the slug."""
    slug = _slug(name)
    if not slug:
        return None
    data = _load_json(_teams_path(data_dir))
    teams = data.setdefault("teams", {})
    teams[slug] = {
        "name": name.strip()[:60],
        "members": sorted({m.strip().lower() for m in members if m.strip()}),
    }
    _save_json(_teams_path(data_dir), data)
    return slug


def delete_team(data_dir: Path, slug: str) -> bool:
    data = _load_json(_teams_path(data_dir))
    removed = data.get("teams", {}).pop(slug, None) is not None
    if removed:
        _save_json(_teams_path(data_dir), data)
    return removed


# --- per-meeting grants (storage in fred_live_worker.meeting_access) -----------

def grant_team(data_dir: Path, meeting_id: str, team_slug: str, granted_by: str) -> int:
    """SNAPSHOT team share: resolve members NOW, store individual grants."""
    team = load_teams(data_dir).get(team_slug)
    if not team:
        return 0
    return grant(
        data_dir, meeting_id, team.get("members", []), granted_by, via=f"team:{team_slug}"
    )
