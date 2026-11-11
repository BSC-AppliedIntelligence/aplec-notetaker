"""Read-model over the worker's on-disk runs (data/meetings/<id>/latest.json).

No database: the run store is the source of truth. Scans are cached per-file by
mtime, so the hot path is a readdir + dict lookups — fine for thousands of runs.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MeetingCard:
    meeting_id: str
    vexa_user_id: int | None
    title: str
    meet_code: str | None
    start_time: datetime | None
    duration_seconds: float | None
    quality: str
    summary: str
    languages: list[str] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)
    # "ready" | "processing" (newest run still in flight) | "failed" (newest
    # run wrote error.json). A failed meeting may still have an older result.
    status: str = "ready"
    error_message: str | None = None
    has_result: bool = True
    # Vexa platform id: "google_meet" | "zoom" | "teams" (future) | None.
    platform: str | None = None

    @property
    def duration_label(self) -> str:
        if not self.duration_seconds:
            return ""
        m = int(self.duration_seconds // 60)
        if m >= 60:
            return f"{m // 60}h {m % 60:02d}min"
        return f"{m}min" if m else f"{int(self.duration_seconds)}s"

    transcript_excerpt: str = ""

    @property
    def search_blob(self) -> str:
        return " ".join(
            [
                self.title,
                self.meet_code or "",
                (self.platform or "").replace("_", " "),
                self.summary,
                " ".join(self.speakers),
                self.transcript_excerpt,
            ]
        ).lower()


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    # Vexa stamps naive UTC; normalize so naive and aware values sort together.
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def load_titles(data_dir: Path) -> dict:
    """meet_code -> {title, ts}, written by the autojoin watcher at launch."""
    path = data_dir / "meet_titles.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


class MeetingIndex:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._cache: dict[str, tuple[tuple, MeetingCard]] = {}

    @property
    def meetings_dir(self) -> Path:
        return self.data_dir / "meetings"

    def _parse_card(self, meeting_id: str, latest: dict, titles: dict) -> MeetingCard:
        meeting = latest.get("meeting") or {}
        transcript = latest.get("transcript") or {}
        insights = latest.get("insights") or {}

        meet_code = meeting.get("native_meeting_id")
        title = None
        if meet_code and meet_code in titles:
            title = (titles[meet_code] or {}).get("title")
        title = title or meet_code or f"Meeting {meeting_id}"

        segments = transcript.get("segments") or []
        languages: list[str] = []
        for seg in segments:
            lang = seg.get("language")
            if lang and lang not in languages:
                languages.append(lang)
        if not languages and transcript.get("language"):
            languages = [transcript["language"]]

        speaker_names = transcript.get("speaker_names") or {}
        speakers = list(dict.fromkeys(speaker_names.values()))
        if not speakers:
            raw = {s.get("speaker") for s in segments if s.get("speaker")}
            speakers = sorted(raw)

        user_id = meeting.get("user_id")
        try:
            user_id = int(user_id) if user_id is not None else None
        except (TypeError, ValueError):
            user_id = None

        return MeetingCard(
            meeting_id=meeting_id,
            vexa_user_id=user_id,
            title=title,
            meet_code=meet_code,
            start_time=_parse_dt(meeting.get("start_time")),
            duration_seconds=meeting.get("duration_seconds")
            or _duration_from_times(meeting),
            quality=latest.get("quality") or "ok",
            summary=(insights.get("summary") or "").strip(),
            # Raw ISO codes; templates format them (flag emoji + label).
            languages=languages,
            speakers=speakers,
            transcript_excerpt=(transcript.get("text") or "")[:20000],
            platform=meeting.get("platform"),
        )

    @staticmethod
    def _newest_run_state(entry: Path) -> tuple[str, Path | None]:
        """('done'|'failed'|'processing'|'none', newest run dir). A run is in
        flight when hook.json exists without result/error; stale in-flight dirs
        (>24h, e.g. worker died and never recovered) are not surfaced."""
        run_dirs = sorted(d for d in entry.iterdir() if d.is_dir())
        if not run_dirs:
            return "none", None
        newest = run_dirs[-1]
        if (newest / "result.json").is_file():
            return "done", newest
        if (newest / "error.json").is_file():
            return "failed", newest
        hook = newest / "hook.json"
        if hook.is_file():
            try:
                age = datetime.now(timezone.utc).timestamp() - hook.stat().st_mtime
            except OSError:
                return "none", newest
            return ("processing", newest) if age < 24 * 3600 else ("none", newest)
        return "none", newest

    def _card_from_hook(self, meeting_id: str, run_dir: Path, titles: dict) -> MeetingCard | None:
        """Minimal card for a meeting that has no result yet (processing/failed
        before the first successful run)."""
        try:
            hook = json.loads((run_dir / "hook.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return self._parse_card(meeting_id, {"meeting": hook.get("meeting") or {}}, titles)

    @staticmethod
    def _error_message(run_dir: Path) -> str | None:
        try:
            err = json.loads((run_dir / "error.json").read_text(encoding="utf-8"))
            msg = f"{err.get('error_type', 'Error')}: {err.get('message', '')}".strip(": ")
            return msg[:300]
        except (OSError, json.JSONDecodeError):
            return None

    def list_cards(self) -> list[MeetingCard]:
        cards: list[MeetingCard] = []
        titles = load_titles(self.data_dir)
        root = self.meetings_dir
        if not root.is_dir():
            return []
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            try:
                latest_path = entry / "latest.json"
                has_latest = latest_path.is_file()
                state, newest_run = self._newest_run_state(entry)

                signature = (
                    latest_path.stat().st_mtime if has_latest else 0.0,
                    state,
                    newest_run.name if newest_run else "",
                )
                cached = self._cache.get(entry.name)
                if cached and cached[0] == signature:
                    cards.append(cached[1])
                    continue

                card: MeetingCard | None = None
                if has_latest:
                    latest = json.loads(latest_path.read_text(encoding="utf-8"))
                    card = self._parse_card(entry.name, latest, titles)
                elif newest_run is not None and state in ("processing", "failed"):
                    card = self._card_from_hook(entry.name, newest_run, titles)
                    if card:
                        card.has_result = False
                if card is None:
                    continue

                if state == "processing":
                    card.status = "processing"
                elif state == "failed":
                    card.status = "failed"
                    card.error_message = self._error_message(newest_run) if newest_run else None

                self._cache[entry.name] = (signature, card)
                cards.append(card)
            except (OSError, json.JSONDecodeError, ValueError):
                logger.warning("Skipping unreadable meeting dir %s", entry, exc_info=True)
        cards.sort(key=lambda c: (c.start_time is not None, c.start_time), reverse=True)
        return cards

    def load_latest(self, meeting_id: str) -> dict | None:
        path = self.meetings_dir / meeting_id / "latest.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None


def _duration_from_times(meeting: dict) -> float | None:
    start = _parse_dt(meeting.get("start_time"))
    end = _parse_dt(meeting.get("end_time"))
    if start and end:
        return max(0.0, (end - start).total_seconds())
    return None


def visible_cards(cards: list[MeetingCard], vexa_user_id: int, is_admin: bool) -> list[MeetingCard]:
    if is_admin:
        return cards
    return [c for c in cards if c.vexa_user_id == vexa_user_id]


def can_view(latest: dict, vexa_user_id: int, is_admin: bool) -> bool:
    if is_admin:
        return True
    owner = (latest.get("meeting") or {}).get("user_id")
    try:
        return owner is not None and int(owner) == vexa_user_id
    except (TypeError, ValueError):
        return False
