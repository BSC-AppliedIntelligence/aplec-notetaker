"""Meeting chat: messages typed in the meeting's own chat panel.

The browser extension reads them from the meeting page (Google Meet, Zoom web)
and uploads them next to the speaker events. Everything here treats them as
UNTRUSTED input — any participant can type anything — so text is bounded,
links are reduced to plain http(s) URLs, and nothing from a message is ever
used as markup.

Chat stays separate from the spoken transcript end to end: it never passes
through the ASR filters (there is no audio to have hallucinated), and a sender
who only typed must never reach the diarization floor, which counts distinct
speaking identities — the same failure the `devices` DOM artifact caused.
"""
from __future__ import annotations

import re
from urllib.parse import urlsplit

from .schemas import ChatMessage, TranscriptResult
from .speaker_merge import render_speaker_transcript

MAX_CHAT_MESSAGES = 2000
MAX_SENDER_CHARS = 120
MAX_TEXT_CHARS = 4000
MAX_LINKS_PER_MESSAGE = 20
MAX_LINK_CHARS = 2000

_URL_RE = re.compile(r"https?://[^\s<>\"'`]+", re.IGNORECASE)
_TRAILING_PUNCT = ".,;:!?)]}>»”’'\""


def safe_link(url) -> str | None:
    """The URL if it is a plain http(s) link with a host, else None.

    Anything else (javascript:, data:, mailto:, relative paths) is dropped
    rather than repaired: these links end up as clickable anchors on the
    meeting page, viewed by people who did not write them."""
    if not isinstance(url, str):
        return None
    url = url.strip().rstrip(_TRAILING_PUNCT)
    if not url or len(url) > MAX_LINK_CHARS:
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        return None
    if any(ch.isspace() or ord(ch) < 32 for ch in url):
        return None
    return url


def links_in(text: str) -> list[str]:
    """http(s) URLs written in the text itself (platforms do not always turn
    them into anchors)."""
    found = []
    for match in _URL_RE.finditer(text or ""):
        link = safe_link(match.group(0))
        if link:
            found.append(link)
    return found


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def chat_from_hook(raw) -> list[ChatMessage]:
    """Validate the `chat_messages` list carried by a capture's payload."""
    messages: list[ChatMessage] = []
    if not isinstance(raw, list):
        return messages
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()[:MAX_TEXT_CHARS]
        if not text:
            continue
        sender = str(item.get("sender") or "").strip()[:MAX_SENDER_CHARS] or "Unknown"
        rel = item.get("relative_timestamp_ms")
        offset = max(0.0, float(rel) / 1000.0) if isinstance(rel, (int, float)) else None
        given = item.get("links") if isinstance(item.get("links"), list) else []
        links = _dedup([l for l in (safe_link(u) for u in given) if l] + links_in(text))
        messages.append(
            ChatMessage(sender=sender, text=text, offset_seconds=offset,
                        links=links[:MAX_LINKS_PER_MESSAGE])
        )
    messages.sort(key=lambda m: (m.offset_seconds is None, m.offset_seconds or 0.0))
    return messages[:MAX_CHAT_MESSAGES]


def attach_chat(transcript: TranscriptResult, meeting_data: dict | None) -> TranscriptResult:
    """Copy the payload's chat onto the transcript. A transcript that already
    carries chat keeps it (reprocessing an old run must not lose it)."""
    if transcript.chat_messages:
        return transcript
    messages = chat_from_hook((meeting_data or {}).get("chat_messages"))
    if not messages:
        return transcript
    return transcript.model_copy(update={"chat_messages": messages})


def fmt_offset(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def render_chat(messages: list[ChatMessage]) -> str:
    return "\n".join(f"[{fmt_offset(m.offset_seconds)}] {m.sender}: {m.text}" for m in messages)


def transcript_for_llm(transcript: TranscriptResult) -> str:
    """What the summarizer and Ask read: the spoken transcript, then the chat
    as a separately labelled block (typed, not spoken — so a link or a number
    pasted in the chat can be quoted, and is never mistaken for speech)."""
    has_speakers = any(segment.speaker for segment in transcript.segments)
    speech = render_speaker_transcript(transcript.segments) if has_speakers else transcript.text
    chat = render_chat(transcript.chat_messages)
    if not chat:
        return speech
    block = f"Meeting chat (messages typed by participants, not spoken):\n{chat}"
    return f"{speech}\n\n{block}" if speech else block


def shared_links(messages: list[ChatMessage]) -> list[dict]:
    """Every distinct link posted in the chat, first poster first."""
    seen: set[str] = set()
    out = []
    for m in messages:
        for link in m.links:
            if link in seen:
                continue
            seen.add(link)
            out.append({"url": link, "sender": m.sender, "offset_seconds": m.offset_seconds})
    return out
