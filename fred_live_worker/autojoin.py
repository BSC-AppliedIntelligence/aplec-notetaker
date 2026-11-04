"""Calendar auto-join watcher.

Polls Google Calendar and launches a Vexa recording bot for every upcoming
Google Meet event. Runs as its own process (scripts/run_autojoin.sh) so worker
restarts don't interrupt it.

Usage:
    python -m fred_live_worker.autojoin            # loop forever
    python -m fred_live_worker.autojoin --once     # single poll (for testing)

Decision logic (pure functions, unit-tested):
- join when now >= start - lead AND now <= end (covers in-progress meetings)
- skip: all-day events, events the user declined, events without a Meet link,
  meetings already requested (state file data/autojoin_state.json).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from .config import Settings, get_settings

logger = logging.getLogger(__name__)

MEET_CODE_RE = re.compile(r"meet\.google\.com/([a-z]{3}-[a-z]{4}-[a-z]{3})")
STATE_MAX_AGE_DAYS = 7


def extract_meet_code(event: dict) -> str | None:
    """Pull the Meet code (xxx-yyyy-zzz) from conferenceData, hangoutLink,
    location or description — in that order of reliability."""
    conference = event.get("conferenceData") or {}
    for entry_point in conference.get("entryPoints") or []:
        if entry_point.get("entryPointType") == "video":
            match = MEET_CODE_RE.search(entry_point.get("uri") or "")
            if match:
                return match.group(1)

    for field in ("hangoutLink", "location", "description"):
        match = MEET_CODE_RE.search(str(event.get(field) or ""))
        if match:
            return match.group(1)

    return None


def _parse_event_time(value: dict) -> datetime | None:
    stamp = value.get("dateTime")
    if not stamp:
        return None  # all-day events carry only "date"
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def is_declined_by_self(event: dict) -> bool:
    for attendee in event.get("attendees") or []:
        if attendee.get("self") and attendee.get("responseStatus") == "declined":
            return True
    return False


def should_join(
    event: dict,
    now: datetime,
    lead_seconds: int,
    already_requested: set[str],
) -> str | None:
    """Return the Meet code to join, or None."""
    code = extract_meet_code(event)
    if not code or code in already_requested:
        return None
    if event.get("status") == "cancelled" or is_declined_by_self(event):
        return None

    start = _parse_event_time(event.get("start") or {})
    if start is None:
        return None  # all-day
    end = _parse_event_time(event.get("end") or {}) or (start + timedelta(hours=1))

    if now < start - timedelta(seconds=lead_seconds):
        return None  # too early
    if now > end:
        return None  # over
    return code


def should_invite(
    event: dict,
    bot_email: str,
    already_invited: set[str],
) -> str | None:
    """Return the event id if the bot's Google account should be added as an
    attendee, or None. Runs over the full poll lookahead (not just the join lead)
    so the invite lands well before the bot knocks — an invited signed-in account
    joins without Google's "possibly risky" label."""
    event_id = event.get("id")
    if not event_id or event_id in already_invited:
        return None
    if not extract_meet_code(event):
        return None
    if event.get("status") == "cancelled" or is_declined_by_self(event):
        return None
    if _parse_event_time(event.get("start") or {}) is None:
        return None  # all-day
    for attendee in event.get("attendees") or []:
        if (attendee.get("email") or "").lower() == bot_email.lower():
            return None  # already invited
    return event_id


class AutojoinState:
    """Meet codes we launched a bot for + event ids we invited the bot account to,
    persisted across restarts. Migrates the legacy flat {code: ts} shape."""

    def __init__(self, path: Path):
        self.path = path
        self._joined: dict[str, str] = {}
        self._invited: dict[str, str] = {}
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if "joined" in raw or "invited" in raw:
                    self._joined = dict(raw.get("joined") or {})
                    self._invited = dict(raw.get("invited") or {})
                else:  # legacy flat shape: everything was a joined meet code
                    self._joined = dict(raw)
            except (OSError, json.JSONDecodeError):
                logger.warning("Could not read autojoin state at %s; starting fresh", path)

    def codes(self) -> set[str]:
        return set(self._joined)

    def invited_ids(self) -> set[str]:
        return set(self._invited)

    @staticmethod
    def _prune(entries: dict[str, str], now: datetime) -> dict[str, str]:
        cutoff = now - timedelta(days=STATE_MAX_AGE_DAYS)
        return {k: ts for k, ts in entries.items() if datetime.fromisoformat(ts) > cutoff}

    def _write(self, now: datetime) -> None:
        self._joined = self._prune(self._joined, now)
        self._invited = self._prune(self._invited, now)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"joined": self._joined, "invited": self._invited}, indent=2),
            encoding="utf-8",
        )

    def mark(self, code: str, now: datetime) -> None:
        self._joined[code] = now.isoformat()
        self._write(now)

    def mark_invited(self, event_id: str, now: datetime) -> None:
        self._invited[event_id] = now.isoformat()
        self._write(now)


def record_meet_title(data_dir: Path, code: str, title: str | None, now: datetime) -> None:
    """Persist meet_code -> calendar event title (data/meet_titles.json) so the
    web frontend can show real meeting names — Vexa only knows the meet code.
    Best-effort: failures never block a launch. Entries older than 120 days are
    pruned on write."""
    if not title:
        return
    path = data_dir / "meet_titles.json"
    try:
        try:
            titles = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(titles, dict):
                titles = {}
        except (OSError, json.JSONDecodeError):
            titles = {}
        titles[code] = {"title": str(title)[:200], "ts": now.isoformat()}
        cutoff = now - timedelta(days=120)
        for key in list(titles.keys()):
            try:
                if datetime.fromisoformat(titles[key]["ts"]) < cutoff:
                    del titles[key]
            except (KeyError, TypeError, ValueError):
                del titles[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(titles, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        logger.warning("Could not record meeting title for %s", code, exc_info=True)


def calendar_owner_email(token_path: str | Path) -> str | None:
    """gcal_token_<email>.json -> email; None for the legacy operator token."""
    name = Path(token_path).name
    if name.startswith("gcal_token_") and name.endswith(".json"):
        candidate = name[len("gcal_token_"):-len(".json")]
        if "@" in candidate:
            return candidate
    return None


def passes_autojoin_policy(event: dict, policy: str) -> bool:
    """Does this event clear the calendar owner's auto-join policy?

    - "off": nothing does (manual/extension capture only).
    - "all": everything the join rules already allow.
    - "accepted": only meetings the owner organized or explicitly accepted —
      an invite sitting on the calendar as needsAction/tentative is the
      'team meeting I'm not attending' case and gets no uninvited notetaker.
    """
    if policy == "off":
        return False
    if policy == "all":
        return True
    if (event.get("organizer") or {}).get("self"):
        return True
    for attendee in event.get("attendees") or []:
        if attendee.get("self"):
            return attendee.get("responseStatus") == "accepted"
    # No attendee list and not the organizer: a plain personal event with a
    # Meet link (self-created without guests) — the owner's own meeting.
    return not event.get("attendees") and not event.get("organizer")


def bot_launch_payload(settings: Settings, meet_code: str) -> dict:
    """POST /bots body for an autojoin launch."""
    payload = {
        "platform": "google_meet",
        "native_meeting_id": meet_code,
        "bot_name": settings.autojoin_bot_name,
        "recording_enabled": True,
        "transcribe_enabled": False,
        # The bot joins lead_seconds early, so Vexa's 120s no_one_joined default
        # would expire ~30s after the scheduled start. Wait through the lead
        # plus a no-show grace for late humans.
        "automatic_leave": {
            "no_one_joined_timeout": (
                settings.autojoin_lead_seconds + settings.autojoin_no_show_grace_seconds
            )
            * 1000,
            # Guests can't "end for everyone"; when the last human leaves this
            # is what ends the meeting (Vexa default: 15 min of silent lingering).
            "max_time_left_alone": settings.autojoin_alone_timeout_seconds * 1000,
        },
    }
    if settings.autojoin_authenticated:
        payload["authenticated"] = True
    return payload


def discover_calendar_tokens(settings: Settings) -> list[Path]:
    """All connected calendars: per-user tokens written by the web app
    (data/credentials/gcal_token_*.json) plus the legacy env-pointed file.
    Re-run every poll so a calendar connected via the web UI is picked up
    without a watcher restart."""
    tokens: dict[str, Path] = {}
    cred_dir = settings.data_dir / "credentials"
    if cred_dir.is_dir():
        for path in sorted(cred_dir.glob("gcal_token_*.json")):
            tokens[str(path.resolve())] = path
    if settings.gcal_credentials and Path(settings.gcal_credentials).is_file():
        path = Path(settings.gcal_credentials)
        tokens.setdefault(str(path.resolve()), path)
    return list(tokens.values())


class AutojoinWatcher:
    def __init__(self, settings: Settings):
        if not settings.vexa_api_key:
            raise SystemExit("VEXA_API_KEY is not set (user-scoped key required to launch bots)")
        if not settings.vexa_api_base_url:
            raise SystemExit("VEXA_API_BASE_URL is not set")
        self.settings = settings
        self.state = AutojoinState(settings.data_dir / "autojoin_state.json")
        self._services: dict[str, object] = {}

    def _calendar_service(self, token_path: str):
        if token_path not in self._services:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build

            creds = Credentials.from_authorized_user_file(token_path)
            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
            self._services[token_path] = build(
                "calendar", "v3", credentials=creds, cache_discovery=False
            )
        return self._services[token_path]

    def upcoming_events_all(self, now: datetime) -> list[tuple[str, dict]]:
        """(token_path, event) for every connected calendar. One broken
        calendar (revoked token etc.) never blocks the others."""
        window_start = (now - timedelta(minutes=5)).isoformat()
        window_end = (now + timedelta(minutes=15)).isoformat()
        out: list[tuple[str, dict]] = []
        for token in discover_calendar_tokens(self.settings):
            token_path = str(token)
            try:
                response = (
                    self._calendar_service(token_path)
                    .events()
                    .list(
                        calendarId="primary",
                        timeMin=window_start,
                        timeMax=window_end,
                        singleEvents=True,
                        orderBy="startTime",
                        maxResults=20,
                    )
                    .execute()
                )
            except Exception:
                self._services.pop(token_path, None)  # force rebuild next poll
                logger.exception("Calendar poll failed for %s (continuing with others)", token.name)
                continue
            out.extend((token_path, event) for event in (response.get("items") or []))
        return out

    def invite_bot_to_event(self, token_path: str, event: dict) -> bool:
        """Add the bot's Google account as an attendee (sendUpdates=none so no
        notification emails). Invited + signed-in = no 'possibly risky' label.
        Patched via the calendar the event was found on."""
        attendees = list(event.get("attendees") or [])
        attendees.append({"email": self.settings.bot_google_account, "responseStatus": "accepted"})
        (
            self._calendar_service(token_path)
            .events()
            .patch(
                calendarId="primary",
                eventId=event["id"],
                body={"attendees": attendees},
                sendUpdates="none",
            )
            .execute()
        )
        return True

    def api_key_for_calendar(self, token_path: str) -> str | None:
        """The Vexa API key the bot launches with — the CALENDAR OWNER's key,
        so the meeting belongs to that user (in Vexa's DB and therefore in the
        web app's ownership check). The legacy operator token file carries no
        email and uses the env VEXA_API_KEY, as before. A per-user calendar
        whose owner has no stored key is SKIPPED loudly: falling back to the
        operator key would silently record the meeting under the wrong
        account, which is exactly the bug this method exists to end."""
        from .users import load_vexa_keys

        email = calendar_owner_email(token_path)
        if email is None:
            return self.settings.vexa_api_key.get_secret_value()
        key = load_vexa_keys(self.settings.data_dir).get(email.lower())
        if key:
            return key
        logger.error(
            "Calendar %s is connected but %s has no Vexa API key "
            "(run: python -m fred_live_worker.onboard --email %s); NOT launching",
            Path(token_path).name, email, email,
        )
        return None

    async def launch_bot(self, meet_code: str, api_key: str) -> bool:
        payload = bot_launch_payload(self.settings, meet_code)
        headers = {"X-API-Key": api_key}
        url = f"{self.settings.vexa_api_base_url.rstrip('/')}/bots"
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, json=payload, headers=headers)
        if response.status_code in (200, 201):
            logger.info("Launched bot for %s", meet_code)
            return True
        logger.error("Bot launch for %s failed: HTTP %s %s", meet_code, response.status_code, response.text[:200])
        return False

    async def poll_once(self) -> None:
        now = datetime.now(timezone.utc)
        events = await asyncio.to_thread(self.upcoming_events_all, now)

        # Per-calendar auto-join policy (user preference; legacy operator token
        # defaults to "all"). Filtered before BOTH passes so Fred neither
        # invites itself to nor joins meetings the owner didn't opt into.
        from .user_prefs import get_autojoin_policy

        policies: dict[str, str] = {}
        filtered: list[tuple[str, dict]] = []
        for token_path, event in events:
            email = calendar_owner_email(token_path)
            if token_path not in policies:
                policies[token_path] = (
                    get_autojoin_policy(self.settings.data_dir, email) if email else "all"
                )
            if passes_autojoin_policy(event, policies[token_path]):
                filtered.append((token_path, event))
            else:
                logger.debug(
                    "Policy %r skips event %r on %s",
                    policies[token_path], event.get("summary"), Path(token_path).name,
                )
        events = filtered

        # Invite pass: runs over the whole lookahead so the invite lands well
        # before the bot knocks. Failures never block the join pass. Shared
        # events appearing on several calendars dedup via invited_ids/codes.
        if self.settings.autojoin_invite_bot and self.settings.bot_google_account:
            for token_path, event in events:
                event_id = should_invite(event, self.settings.bot_google_account, self.state.invited_ids())
                if event_id is None:
                    continue
                try:
                    await asyncio.to_thread(self.invite_bot_to_event, token_path, event)
                    logger.info("Invited %s to event %r", self.settings.bot_google_account, event.get("summary"))
                    self.state.mark_invited(event_id, now)
                except Exception:
                    logger.exception("Could not invite bot to event %r (continuing)", event.get("summary"))

        from .capture import captured_codes

        claimed = self.state.codes() | captured_codes(self.settings.data_dir)
        for token_path, event in events:
            code = should_join(event, now, self.settings.autojoin_lead_seconds, claimed)
            if code is None:
                continue
            api_key = self.api_key_for_calendar(token_path)
            if api_key is None:
                continue  # already logged: connected calendar, un-onboarded user
            logger.info("Event %r starts soon; launching bot for %s", event.get("summary"), code)
            if await self.launch_bot(code, api_key):
                self.state.mark(code, now)
                record_meet_title(self.settings.data_dir, code, event.get("summary"), now)

    async def run_forever(self) -> None:
        tokens = discover_calendar_tokens(self.settings)
        logger.info(
            "Autojoin watcher started (lead=%ss, poll=%ss, %d connected calendar(s): %s)",
            self.settings.autojoin_lead_seconds,
            self.settings.autojoin_poll_seconds,
            len(tokens),
            ", ".join(t.name for t in tokens) or "none yet — connect one via the web UI",
        )
        while True:
            await self.poll_once()
            await asyncio.sleep(self.settings.autojoin_poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Poll once and exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    watcher = AutojoinWatcher(get_settings())
    if args.once:
        asyncio.run(watcher.poll_once())
    else:
        asyncio.run(watcher.run_forever())


if __name__ == "__main__":
    main()
