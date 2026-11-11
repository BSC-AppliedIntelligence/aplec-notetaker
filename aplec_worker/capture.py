"""Browser-extension capture ingest (the no-bot mode).

The Meet Chrome extension records tab audio + mic locally in the user's
browser and streams webm chunks here; on finish we synthesize the same
meeting-completed payload Vexa's webhook would deliver and run the normal
pipeline — diarization, speaker naming from the extension's DOM events,
multilingual ASR, summary, web UI. No bot in the call.

Auth: a per-user capture token minted by the web app, signed with the shared
APLEC_WEB_SESSION_SECRET (both apps read .env.worker). Long-lived; pasted
once into the extension's options.

Sessions live at data/captures/<session_id>/ (meta.json + events.jsonl) with
audio appended to recordings/captures/<session_id>/master.webm — inside
APLEC_RECORDINGS_DIR so the normal path resolver accepts it.

Same-meeting resume: stopping and restarting the recording of the SAME meet
code by the same user within APLEC_CAPTURE_RESUME_GAP_MINUTES (default 30)
reuses the session instead of creating a new meeting. Each restart appends a
"part" (its own webm file — MediaRecorder output can't be byte-appended across
recorder instances) with an offset equal to the summed duration of the earlier
parts; speaker-event offsets are shifted by it. finish concatenates the parts
(ffmpeg concat -> 16k mono wav) and reprocesses the SAME meeting id, so the
history shows one meeting whose notes absorb every part. The gap keeps
recurring meetings that reuse one link (e.g. weeklies) as separate meetings.
Platforms that expose no meeting id (Teams v2 rewrites the join URL away as
soon as the call starts) fall back to the meeting TITLE as identity — see
resume_key().
"""
from __future__ import annotations

import json
import logging
import re
import secrets
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

CAPTURE_TOKEN_SALT = "aplec-capture"
# Left and right of what the extension records (v0.3.14+).
MIC_AND_TAB = ["mic", "tab"]
MAX_CHUNK_BYTES = 32 * 1024 * 1024
MAX_EVENTS_PER_POST = 500
MAX_NOTICE_TEXT_CHARS = 500
MAX_NOTICE_EVENTS = 50
# sent: posted in the meeting chat · failed: tried and could not post ·
# disabled: the recorder turned the automatic notice off · unsupported: this
# platform/page offers no chat the extension can post to
NOTICE_STATUSES = ("sent", "failed", "disabled", "unsupported")
_SESSION_ID_RE = re.compile(r"^cap-[a-f0-9]{12}$")


def mint_capture_token(secret: str, vexa_user_id: int, email: str) -> str:
    from itsdangerous import URLSafeSerializer

    return URLSafeSerializer(secret, salt=CAPTURE_TOKEN_SALT).dumps(
        {"uid": int(vexa_user_id), "email": email}
    )


def verify_capture_token(secret: str, token: str) -> dict | None:
    from itsdangerous import BadSignature, URLSafeSerializer

    try:
        data = URLSafeSerializer(secret, salt=CAPTURE_TOKEN_SALT).loads(token)
    except BadSignature:
        return None
    if not isinstance(data, dict) or "uid" not in data:
        return None
    return data


def allowed_capture_user(web_users: str, user: dict, data_dir: Path | None = None) -> bool:
    """True if the token's (email, uid) still matches the allowlist (env
    APLEC_WEB_USERS merged with data/users.json when data_dir is given),
    mirroring the web app's user_map so a removed email revokes the capture
    token too. Matching on email AND uid means a re-created account (same
    email, new uid) also invalidates the old token."""
    from .users import merged_users, parse_env_users

    email = str(user.get("email") or "").strip().lower()
    if not email:
        return False
    allowed = merged_users(web_users, data_dir) if data_dir else parse_env_users(web_users)
    entry = allowed.get(email)
    if entry is None:
        return False
    try:
        return entry.vexa_user_id == int(user.get("uid"))
    except (TypeError, ValueError):
        return False


# Titles too generic to identify a meeting: a platform default would merge
# unrelated sittings into one history card.
GENERIC_TITLES = {
    "meeting", "meet", "google meet", "call", "untitled",
    "microsoft teams", "teams", "teams meeting", "microsoft teams meeting",
    "zoom", "zoom meeting", "personal meeting room",
}


def _normalize_title(title: str | None) -> str:
    t = re.sub(r"^\(\d+\)\s*", "", str(title or "").strip())  # Teams unread prefix
    return re.sub(r"\s+", " ", t).strip().lower()


def resume_key(platform: str | None, native_meeting_id: str | None, title: str | None) -> str | None:
    """What makes two starts 'the same meeting' for the resume merge.

    Normally the platform's own meeting id. Teams v2 has none to give: it
    rewrites the join URL (the only place the thread id appears) as soon as
    the call starts, so a Teams capture arrives with an empty id and leaving
    and rejoining used to produce two separate meetings. There the TITLE
    carries the identity — it is stable across a rejoin within the gap.
    None means "never resume": no id and no title specific enough to trust."""
    platform = platform or "google_meet"
    if native_meeting_id:
        return f"id:{platform}:{native_meeting_id}"
    normalized = _normalize_title(title)
    if len(normalized) < 4 or normalized in GENERIC_TITLES:
        return None
    return f"title:{platform}:{normalized}"


class CaptureStore:
    def __init__(self, settings):
        self.settings = settings

    def _captures_dir(self) -> Path:
        return self.settings.data_dir / "captures"

    def session_dir(self, session_id: str) -> Path:
        if not _SESSION_ID_RE.fullmatch(session_id):
            raise ValueError("Invalid session id")
        return self._captures_dir() / session_id

    def recording_dir(self, session_id: str) -> Path:
        return self.settings.recordings_dir / "captures" / session_id

    def audio_path(self, session_id: str) -> Path:
        return self.recording_dir(session_id) / "master.webm"

    @staticmethod
    def _parts(meta: dict) -> list[dict]:
        parts = meta.get("parts")
        if isinstance(parts, list) and parts:
            return parts
        # legacy single-part session
        return [{"file": "master.webm", "started_at_ms": int(meta.get("started_at_ms") or 0), "offset_ms": 0}]

    def _write_meta(self, session_id: str, meta: dict) -> None:
        (self.session_dir(session_id) / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )

    def start(
        self,
        user: dict,
        platform: str,
        native_meeting_id: str,
        title: str | None,
        part_nonce: int | None = None,
        ext_version: str | None = None,
        notice_enabled: bool | None = None,
        notice_text: str | None = None,
        audio_channels: list[str] | None = None,
    ) -> tuple[str, bool]:
        """Returns (session_id, resumed). A start for the same user + meet code
        within the resume gap continues the existing session as a new part,
        so restarts of one sitting stay ONE meeting in the history.

        part_nonce (v0.3.8 clients): minted by the extension BEFORE its
        recorder starts and stamped on the part created here; every chunk from
        that recorder carries it. It is what lets append_chunk_dedup drop a
        chunk from an OLD recorder still draining its retry buffer after a
        resume created a new part — without it those bytes would append into
        the new part's webm and corrupt it.

        audio_channels (v0.3.14 clients): what the recording's channels hold; any
        other value than the one layout that exists counts as a mono recording."""
        split = audio_channels == MIC_AND_TAB
        resumable = self._resumable_session(user, platform, native_meeting_id, title)
        if resumable is not None:
            self._add_part(resumable, part_nonce)
            meta = self.load_meta(resumable)
            dirty = False
            if split and not meta.get("audio_channels"):
                meta["audio_channels"] = list(MIC_AND_TAB)
                dirty = True
            if title and not meta.get("title"):
                meta["title"] = title.strip()[:200] or None
                dirty = True
            if ext_version and meta.get("ext_version") != ext_version:
                # Keep the LATEST build seen: Chrome can update the extension
                # between two parts of the same sitting.
                meta["ext_version"] = str(ext_version)[:32]
                dirty = True
            if notice_enabled is not None:
                # Each part is a new recording start, so its notice setting is
                # recorded too; the per-attempt outcomes accumulate in events.
                notice = meta.get("recording_notice") or {"events": []}
                notice["enabled"] = bool(notice_enabled)
                notice["text"] = (notice_text or "").strip()[:MAX_NOTICE_TEXT_CHARS] or None
                meta["recording_notice"] = notice
                dirty = True
            if dirty:
                self._write_meta(resumable, meta)
            return resumable, True

        session_id = f"cap-{secrets.token_hex(6)}"
        sdir = self.session_dir(session_id)
        sdir.mkdir(parents=True, exist_ok=True)
        now_ms = int(time.time() * 1000)
        first_part = {"file": "master.webm", "started_at_ms": now_ms, "offset_ms": 0}
        if part_nonce is not None:
            first_part["nonce"] = int(part_nonce)
        meta = {
            "session_id": session_id,
            "user_id": int(user["uid"]),
            "email": user.get("email"),
            "platform": platform,
            "native_meeting_id": native_meeting_id,
            "title": (title or "").strip()[:200] or None,
            "started_at_ms": now_ms,
            "parts": [first_part],
            "ext_version": (str(ext_version)[:32] if ext_version else None),
        }
        if split:
            meta["audio_channels"] = list(MIC_AND_TAB)
        if notice_enabled is not None:
            # Absent (not False) for extensions older than the notice feature,
            # so the meeting page can tell "turned off" from "unknown".
            meta["recording_notice"] = {
                "enabled": bool(notice_enabled),
                "text": (notice_text or "").strip()[:MAX_NOTICE_TEXT_CHARS] or None,
                "events": [],
            }
        self._write_meta(session_id, meta)
        audio = self.audio_path(session_id)
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.touch()
        logger.info(
            "Capture %s started by user %s for %s (%s), extension v%s",
            session_id, meta["user_id"], native_meeting_id, meta["title"] or "untitled",
            meta["ext_version"] or "unknown",
        )
        return session_id, False

    def _resumable_session(
        self, user: dict, platform: str, native_meeting_id: str, title: str | None = None
    ) -> str | None:
        gap_minutes = int(getattr(self.settings, "capture_resume_gap_minutes", 0) or 0)
        key = resume_key(platform, native_meeting_id, title)
        if key is None or gap_minutes <= 0:
            return None
        gap_ms = gap_minutes * 60_000
        now_ms = int(time.time() * 1000)
        best: tuple[str, int] | None = None
        for meta_path in self._captures_dir().glob("cap-*/meta.json"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            # Candidate keys are computed from stored fields, so sessions
            # written before this existed match without a migration.
            if (
                resume_key(
                    meta.get("platform"), meta.get("native_meeting_id"), meta.get("title")
                )
                != key
                or int(meta.get("user_id", -1)) != int(user["uid"])
            ):
                continue
            ref = self._last_activity_ms(meta)
            if now_ms - ref <= gap_ms and (best is None or ref > best[1]):
                best = (str(meta["session_id"]), ref)
        return best[0] if best else None

    def _last_activity_ms(self, meta: dict) -> int:
        """Newest sign of life: finish stamp, part-file mtimes, part starts."""
        ref = int(meta.get("finished_at_ms") or 0)
        session_id = str(meta.get("session_id") or "")
        for part in self._parts(meta):
            ref = max(ref, int(part.get("started_at_ms") or 0))
            try:
                path = self.recording_dir(session_id) / str(part.get("file"))
                if path.is_file():
                    ref = max(ref, int(path.stat().st_mtime * 1000))
            except (OSError, ValueError):
                continue
        return max(ref, int(meta.get("started_at_ms") or 0))

    def _add_part(self, session_id: str, part_nonce: int | None = None) -> None:
        meta = self.load_meta(session_id)
        parts = self._parts(meta)
        offset_s = 0.0
        for part in parts:
            path = self.recording_dir(session_id) / str(part["file"])
            if path.is_file() and path.stat().st_size > 0:
                offset_s += self.part_duration_seconds(path)
        new_part = {
            "file": f"master-p{len(parts) + 1}.webm",
            "started_at_ms": int(time.time() * 1000),
            "offset_ms": int(offset_s * 1000),
        }
        if part_nonce is not None:
            new_part["nonce"] = int(part_nonce)
        parts.append(new_part)
        meta["parts"] = parts
        meta.pop("finished_at_ms", None)  # reopened
        self._write_meta(session_id, meta)
        path = self.recording_dir(session_id) / new_part["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        logger.info(
            "Capture %s resumed: part %d starts at offset %.1fs",
            session_id, len(parts), offset_s,
        )

    def part_duration_seconds(self, path: Path) -> float:
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(path)],
                capture_output=True, text=True, check=True, timeout=30,
            ).stdout.strip()
            return max(0.0, float(out))
        except (OSError, ValueError, subprocess.SubprocessError):
            from .audio_io import SAMPLE_RATE, load_waveform_16k

            return len(load_waveform_16k(path)) / SAMPLE_RATE

    def _concat_parts(self, part_files: list[Path], out: Path, channels: int = 1) -> None:
        """Separate MediaRecorder instances can't be byte-concatenated, so we
        splice at the decoded level: ffmpeg concat demuxer -> 16k wav (what the
        ASR consumes anyway), mono unless the recording keeps its channels apart."""
        list_file = out.with_suffix(".parts.txt")
        list_file.write_text(
            "".join(f"file '{p.as_posix()}'\n" for p in part_files), encoding="utf-8"
        )
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
                 "-ac", str(channels), "-ar", "16000", "-c:a", "pcm_s16le", str(out)],
                capture_output=True, check=True, timeout=600,
            )
        finally:
            list_file.unlink(missing_ok=True)

    def load_meta(self, session_id: str) -> dict:
        return json.loads((self.session_dir(session_id) / "meta.json").read_text(encoding="utf-8"))

    def check_owner(self, session_id: str, user: dict) -> dict:
        meta = self.load_meta(session_id)
        if int(meta["user_id"]) != int(user["uid"]):
            raise PermissionError("Session belongs to another user")
        return meta

    def session_bytes(self, session_id: str) -> int:
        """Total uploaded audio across all parts, for the per-session cap."""
        meta = self.load_meta(session_id)
        total = 0
        for part in self._parts(meta):
            path = self.recording_dir(session_id) / str(part["file"])
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
        return total

    def append_chunk(self, session_id: str, data: bytes) -> int:
        """MediaRecorder timeslice chunks of one recorder session concatenate
        into a valid webm stream — same trick as Vexa's chunked upload. Chunks
        always belong to the CURRENT (last) part. Legacy path (pre-v0.3.8
        clients without part/seq): unconditional append, no reopen."""
        meta = self.load_meta(session_id)
        audio = self.recording_dir(session_id) / str(self._parts(meta)[-1]["file"])
        audio.parent.mkdir(parents=True, exist_ok=True)
        with audio.open("ab") as fh:
            fh.write(data)
        return audio.stat().st_size

    def append_chunk_dedup(
        self, session_id: str, data: bytes, part_nonce: int, seq: int
    ) -> tuple[str, int]:
        """At-least-once-safe append for retrying clients. Returns
        (status, total_bytes) with status in {"appended", "stale_part",
        "duplicate"} — non-appended statuses are ACKED to the client (it must
        drop the chunk and move on), never errors.

        * stale_part: the chunk's recorder nonce is not the CURRENT part's —
          an old recorder's retry buffer draining after a same-meeting resume
          created a new part. Appending would write old-part clusters into the
          new part's webm, corrupting it.
        * duplicate: seq <= the part's last appended seq — a retry whose
          previous attempt actually landed but whose response was lost. A
          second append would duplicate ~15s of audio AND (via reopen below)
          could resurrect the duplicate-processing incident after finish.
        * appended: genuinely new bytes. If the session was auto-finished
          meanwhile (janitor, mid-outage), it is REOPENED (finished_at_ms
          cleared) so the eventual real finish processes the FULL audio."""
        meta = self.load_meta(session_id)
        last = self._parts(meta)[-1]
        if int(last.get("nonce") or -1) != int(part_nonce):
            return "stale_part", 0
        if int(seq) <= int(last.get("last_seq", -1)):
            return "duplicate", 0
        total = self.append_chunk(session_id, data)
        # Re-read AFTER the append: a janitor finish_payload interleaving on
        # the event loop must not have its finished_at_ms stamp (or a resume's
        # new part) clobbered by our stale copy of meta.
        meta = self.load_meta(session_id)
        parts = self._parts(meta)
        for part in reversed(parts):
            if int(part.get("nonce") or -1) == int(part_nonce):
                part["last_seq"] = int(seq)
                break
        reopened = meta.pop("finished_at_ms", None) is not None
        self._write_meta(session_id, meta)
        if reopened:
            logger.info(
                "Capture %s reopened: new audio (part %s seq %s) arrived after auto-finish",
                session_id, part_nonce, seq,
            )
        return "appended", total

    def append_events(self, session_id: str, events: list[dict], meta: dict) -> int:
        started_at_ms = int(meta.get("started_at_ms") or 0)
        # Live events always belong to the current part; place them on the
        # concatenated-audio timeline by adding the part's start offset.
        part_offset_ms = int(self._parts(meta)[-1].get("offset_ms") or 0)
        accepted = []
        for event in events[:MAX_EVENTS_PER_POST]:
            if not isinstance(event, dict):
                continue
            etype = event.get("event_type")
            name = str(event.get("participant_name") or "").strip()
            rel = event.get("relative_ts_ms")
            ts = event.get("absolute_ts_ms")
            if etype not in ("SPEAKER_START", "SPEAKER_END") or not name:
                continue
            # Prefer the client-computed offset: it is measured on ONE clock
            # (the browser's) against the actual recorder start. Deriving it
            # here from absolute_ts_ms - started_at_ms mixes the laptop and
            # server clocks; observed skew of minutes zeroed every event.
            if isinstance(rel, (int, float)):
                relative = max(0, int(rel)) + part_offset_ms
            elif isinstance(ts, (int, float)):
                relative = max(0, int(ts - started_at_ms))
            else:
                continue
            accepted.append({
                "event_type": etype,
                "participant_name": name[:120],
                "participant_id": str(event.get("participant_id") or name)[:120],
                "relative_timestamp_ms": relative,
            })
        if accepted:
            with (self.session_dir(session_id) / "events.jsonl").open("a", encoding="utf-8") as fh:
                for event in accepted:
                    fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        return len(accepted)

    def _relative_ms(self, item: dict, meta: dict) -> int | None:
        """Same timeline rule as append_events: the browser-measured offset
        (one clock), shifted onto the concatenated audio by the current part's
        start; absolute-minus-start only as a legacy fallback."""
        rel = item.get("relative_ts_ms")
        if isinstance(rel, (int, float)):
            return max(0, int(rel)) + int(self._parts(meta)[-1].get("offset_ms") or 0)
        ts = item.get("absolute_ts_ms")
        if isinstance(ts, (int, float)):
            return max(0, int(ts - int(meta.get("started_at_ms") or 0)))
        return None

    def append_chat(self, session_id: str, messages: list[dict], meta: dict) -> int:
        """Store chat messages read from the meeting page. Untrusted input:
        bounded, links reduced to http(s), and de-duplicated by the page's
        message key, because the extension re-sends after an outage and the
        platforms re-render the chat list (scrolling, panel reopened)."""
        from .meeting_chat import (
            MAX_CHAT_MESSAGES, MAX_LINKS_PER_MESSAGE, MAX_SENDER_CHARS, MAX_TEXT_CHARS, safe_link,
        )

        existing = self.read_chat(session_id)
        seen = {str(m.get("msg_key")) for m in existing}
        room = MAX_CHAT_MESSAGES - len(existing)
        accepted = []
        for item in messages[:MAX_EVENTS_PER_POST]:
            if room <= 0:
                break
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()[:MAX_TEXT_CHARS]
            if not text:
                continue
            sender = str(item.get("sender") or "").strip()[:MAX_SENDER_CHARS] or "Unknown"
            key = str(item.get("msg_key") or "").strip()[:200] or f"{sender}|{text}"
            if key in seen:
                continue
            relative = self._relative_ms(item, meta)
            raw_links = item.get("links") if isinstance(item.get("links"), list) else []
            links = []
            for url in raw_links[:MAX_LINKS_PER_MESSAGE]:
                link = safe_link(url)
                if link and link not in links:
                    links.append(link)
            seen.add(key)
            room -= 1
            accepted.append({
                "msg_key": key,
                "sender": sender,
                "text": text,
                "links": links,
                "relative_timestamp_ms": relative,
            })
        if accepted:
            with (self.session_dir(session_id) / "chat.jsonl").open("a", encoding="utf-8") as fh:
                for message in accepted:
                    fh.write(json.dumps(message, ensure_ascii=False) + "\n")
        return len(accepted)

    def read_chat(self, session_id: str) -> list[dict]:
        path = self.session_dir(session_id) / "chat.jsonl"
        messages: list[dict] = []
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    messages.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return messages

    def record_notice(self, session_id: str, notice: dict, meta: dict) -> dict:
        """Audit trail of the recording notice posted in the meeting chat: one
        entry per attempt with its outcome (kind "start"; 0.3.12 installs also
        send "rejoin" re-posts, which 0.3.13 removed). Kept in meta.json so it
        travels with the meeting payload and survives reprocessing."""
        status = str(notice.get("status") or "")
        if status not in NOTICE_STATUSES:
            raise ValueError(f"Unknown notice status: {status!r}")
        kind = str(notice.get("kind") or "start")
        entry = {
            "status": status,
            "kind": kind if kind in ("start", "rejoin") else "start",
            "relative_timestamp_ms": self._relative_ms(notice, meta),
            "received_at_ms": int(time.time() * 1000),
        }
        reason = str(notice.get("reason") or "").strip()[:200]
        if reason:
            entry["reason"] = reason
        meta = self.load_meta(session_id)  # fresh: chunk appends write meta too
        container = meta.get("recording_notice") or {"events": []}
        events = container.setdefault("events", [])
        if len(events) < MAX_NOTICE_EVENTS:
            events.append(entry)
        meta["recording_notice"] = container
        self._write_meta(session_id, meta)
        logger.info("Capture %s recording notice: %s (%s)%s", session_id, status, entry["kind"],
                    f" — {reason}" if reason else "")
        return entry

    def stale_unfinished_sessions(self, older_than_minutes: int = 15) -> list[dict]:
        """Sessions never finished (client died, tab killed, stuck REC) whose
        last sign of life is older than the threshold. Live captures upload a
        chunk every 15s, so they always look fresh."""
        cutoff_ms = int(time.time() * 1000) - older_than_minutes * 60_000
        stale = []
        for meta_path in self._captures_dir().glob("cap-*/meta.json"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if meta.get("finished_at_ms"):
                continue
            if self._last_activity_ms(meta) < cutoff_ms:
                stale.append(meta)
        return stale

    def has_audio(self, session_id: str) -> bool:
        meta = self.load_meta(session_id)
        return any(
            (p := self.recording_dir(session_id) / str(part["file"])).is_file() and p.stat().st_size > 0
            for part in self._parts(meta)
        )

    def mark_finished(self, session_id: str) -> None:
        meta = self.load_meta(session_id)
        meta["finished_at_ms"] = int(time.time() * 1000)
        self._write_meta(session_id, meta)

    def append_diagnostics(self, session_id: str, diagnostics: dict) -> None:
        line = json.dumps(diagnostics, ensure_ascii=False, default=str)[:2000]
        logger.info("Capture %s page diagnostics: %s", session_id, line)
        with (self.session_dir(session_id) / "diagnostics.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def read_events(self, session_id: str) -> list[dict]:
        path = self.session_dir(session_id) / "events.jsonl"
        events: list[dict] = []
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return events

    def finish_payload(self, session_id: str) -> dict:
        """Synthesize the meeting-completed payload the pipeline expects.
        Multi-part sessions (same-meeting resume) are concatenated into one
        16k mono wav first; the meeting id stays the session id, so the rerun
        replaces the earlier part-only result in the history."""
        from datetime import datetime, timezone

        meta = self.load_meta(session_id)
        part_files = [
            self.recording_dir(session_id) / str(part["file"])
            for part in self._parts(meta)
        ]
        part_files = [p for p in part_files if p.is_file() and p.stat().st_size > 0]
        if not part_files:
            raise FileNotFoundError("No audio was uploaded for this capture")
        if len(part_files) == 1:
            audio = part_files[0]
        else:
            audio = self.recording_dir(session_id) / "combined.wav"
            self._concat_parts(part_files, audio, channels=2 if meta.get("audio_channels") else 1)
        meta["finished_at_ms"] = int(time.time() * 1000)
        self._write_meta(session_id, meta)
        started = datetime.fromtimestamp(meta["started_at_ms"] / 1000, tz=timezone.utc)
        # end_time comes from the AUDIO, not the wall clock: a finish that
        # arrives late (janitor auto-finish after a crash, a stale client's
        # duplicate) must not inflate the meeting's displayed duration (field:
        # a 59-minute meeting showed 2h21m after a finish 82 minutes late).
        ended = None
        try:
            last = self._parts(meta)[-1]
            last_path = self.recording_dir(session_id) / str(last["file"])
            if last_path.is_file() and last_path.stat().st_size > 0:
                ended = datetime.fromtimestamp(
                    (int(last["started_at_ms"])
                     + int(self.part_duration_seconds(last_path) * 1000)) / 1000,
                    tz=timezone.utc,
                )
        except Exception:
            logger.warning("Could not derive end_time from audio; using now()", exc_info=True)
        if ended is None or ended < started:
            ended = datetime.now(timezone.utc)
        return {
            "event": "meeting.completed",
            "meeting": {
                "id": session_id,
                "user_id": meta["user_id"],
                "platform": meta.get("platform") or "google_meet",
                "native_meeting_id": meta.get("native_meeting_id"),
                "status": "completed",
                "start_time": started.isoformat(),
                "end_time": ended.isoformat(),
                "data": {
                    "source": "browser-extension",
                    "speaker_events": self.read_events(session_id),
                    "chat_messages": self.read_chat(session_id),
                    "recording_notice": meta.get("recording_notice"),
                    "audio_channels": meta.get("audio_channels"),
                    "recordings": [
                        {
                            "id": session_id,
                            "status": "completed",
                            "media_files": [
                                {
                                    "id": 1,
                                    "type": "audio",
                                    "format": audio.suffix.lstrip(".") or "webm",
                                    "storage_path": str(audio),
                                    "file_size_bytes": audio.stat().st_size,
                                }
                            ],
                        }
                    ],
                },
            },
        }


def record_captured_code(data_dir: Path, native_meeting_id: str | None) -> None:
    """Collision guard: tell the calendar watcher this meet code is being
    captured by the extension so the bot doesn't also join it."""
    if not native_meeting_id:
        return
    path = data_dir / "ext_captures.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except (OSError, json.JSONDecodeError):
        data = {}
    from datetime import datetime, timezone

    data[native_meeting_id] = datetime.now(timezone.utc).isoformat()
    # prune entries older than a day
    cutoff = time.time() - 24 * 3600
    for code, iso in list(data.items()):
        try:
            if datetime.fromisoformat(iso).timestamp() < cutoff:
                del data[code]
        except ValueError:
            del data[code]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def captured_codes(data_dir: Path) -> set[str]:
    """Codes currently claimed by extension captures (read by the watcher)."""
    try:
        data = json.loads((data_dir / "ext_captures.json").read_text(encoding="utf-8"))
        return set(data) if isinstance(data, dict) else set()
    except (OSError, json.JSONDecodeError):
        return set()
