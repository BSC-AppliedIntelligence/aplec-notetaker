"""Anti-hallucination text filters for ASR output.

Whisper's observed failure modes on real meetings:

1. Decoder repetition loops: "No, no, no, ..." x14, "Con Ana. Con
   Ana. Con Ana." Boundary-sensitive: the same audio transcribes
   cleanly or loops depending on span edges, so the fix is detect + retry
   (asr.transcribe_block) with these functions as the detector, plus a
   textual collapse as the last-resort safety net.
2. Stock-phrase fabrications from YouTube-ish training data: real speech
   replaced by "¡Suscríbete al canal!". Loud, real audio, so VAD cannot
   prevent it; a per-language blocklist can catch it after the fact.
3. Near-duplicate text on OVERLAPPING diarization turns: both sides of an
   overlap transcribe the same (louder) voice from the mono mix. Dropped by
   time-overlap + token-similarity dedup.

All pure functions; unit-tested without models.
"""
from __future__ import annotations

import functools
import re
import unicodedata

# Known fabrication phrases by language prefix (lowercased, accent-stripped
# substring match). Sources: community-documented Whisper hallucinations
# (Amara/YouTube corpus artifacts) plus phrases observed in our meetings.
FABRICATION_PHRASES: dict[str, tuple[str, ...]] = {
    "es": (
        "suscribete al canal",
        "suscribanse al canal",
        "no olvides suscribirte",
        "subtitulos por la comunidad",
        "subtitulos realizados por",
        "gracias por ver el video",
        "gracias por ver este video",
        "nos vemos en el proximo video",
    ),
    "en": (
        "thanks for watching",
        "thank you for watching",
        "please subscribe",
        "subscribe to the channel",
        # observed concatenated with "thank you for watching" in a real meeting
        "subscribe to my channel",
        "subtitles by the amara",
        "see you in the next video",
    ),
    "pt": (
        "legendas pela comunidade",
        "inscreva-se no canal",
        "obrigado por assistir",
        "ate o proximo video",
    ),
    "ca": (
        "subscriu-te al canal",
        "gracies per veure",
        "subtitols per la comunitat",
    ),
    "de": (
        "danke furs zuschauen",
        "vielen dank furs zuschauen",
        "untertitel der amara",
        "abonniert den kanal",
    ),
    "fr": (
        "merci d'avoir regarde",
        "abonnez-vous",
        "sous-titres realises par",
    ),
    "it": (
        "grazie per aver guardato",
        "iscriviti al canale",
        "sottotitoli creati dalla comunita",
    ),
    # Observed in real meetings (18 surviving copies of the Indonesian one in a
    # single meeting, plus Russian subtitle-credit boilerplate).
    # Whisper emits these in languages NOBODY in the meeting speaks, so the
    # blocklist cannot be limited to the team's working languages.
    "id": (
        "terima kasih telah menonton",
        "jangan lupa subscribe",
    ),
    "ru": (
        "субтитры сделал",
        "субтитры подогнал",
        "редактор субтитров",
        "продолжение следует",
        "субтитры и перевод",
    ),
    "tr": (
        "abone ol",
        "abone olmayi unutmayin",
        "izlediginiz icin tesekkurler",
    ),
}

# Distinctive tokens from the phrases above: words that do not occur in real
# meeting speech. A SHORT segment containing one is a fabrication even when it
# does not contain a full blocklist phrase — field case: "¡Suscríbete!" (13.4s,
# attributed to a real speaker) escaped, because the stored phrase
# "suscribete al canal" is LONGER than the text, and the match was one-way.
# Deliberately NOT here, though both appear in the phrases above: "subscribe",
# which people say about topics, queues and newsletters ("we should
# subscribe to that" was being dropped as a fabrication), and "abone", which is
# also Spanish for "pay/fertilise" ("abone la factura"). Both are only caught as
# part of a full phrase, where the surrounding words disambiguate them.
_FABRICATION_MARKERS: tuple[str, ...] = (
    "suscribete",
    "abonnez",
    "abonniert",
    "iscriviti",
    "inscreva",
    "menonton",
    "субтитры",
)
_MARKER_MAX_TOKENS = 6

# Shortest stretch of repeated words that counts as an overlap echo rather than
# a genuine short interjection by another speaker (see dedup_overlapping_segments).
_MIN_ECHO_TOKENS = 4


def _fold(text: str) -> str:
    """Lowercase and strip accents/punctuation for robust substring matching.

    Hyphens and apostrophes become SPACES, not nothing: joining the parts made
    "Inscreva-se" fold to "inscrevase", so neither a word-level marker nor a
    stored phrase written with a hyphen or apostrophe ("abonnez-vous",
    "merci d'avoir regarde") could ever match.
    """
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[-‐‑‒–—'’`´]", " ", text)
    return re.sub(r"[^\w\s]", "", text)


def is_fabrication(text: str, language: str | None) -> bool:
    """True when the segment is (almost entirely) a known stock fabrication.
    Only short segments qualify: a long genuine sentence that merely CONTAINS
    a phrase must survive.

    Two matches, because the one-way substring test missed both directions of
    real field failures: the phrase inside the segment (with a tight length
    margin, so "Thanks for watching the demo, team" — plausible meeting speech —
    is NOT dropped), and a distinctive marker inside a short segment (so the
    abbreviated "¡Suscríbete!" is)."""
    folded = _fold(text).strip()
    if not folded:
        return False
    phrases = _folded_phrases().get((language or "").split("-")[0], ())
    # check the segment's own language first, then all languages (whisper
    # sometimes mislabels the language of the fabrication itself, and emits
    # phrases in languages nobody in the meeting speaks)
    candidates = phrases + tuple(p for ps in _folded_phrases().values() for p in ps)
    # Coverage, not absolute length: how much of the segment IS blocklist
    # phrasing. A fixed character margin cannot separate "Legendas pela
    # comunidade Amara.org" (the phrase plus a suffix — fabrication) from
    # "Thanks for watching the demo, team" (the phrase plus real content), and
    # it misses two phrases concatenated ("Thank you for watching. Please
    # subscribe to my channel."), which together cover the whole segment.
    covered = bytearray(len(folded))
    for phrase in candidates:
        start = folded.find(phrase)
        while start != -1:
            covered[start : start + len(phrase)] = b"\x01" * len(phrase)
            start = folded.find(phrase, start + 1)
    meaningful = [i for i, ch in enumerate(folded) if not ch.isspace()]
    if meaningful and sum(covered[i] for i in meaningful) / len(meaningful) >= 0.7:
        return True
    tokens = folded.split()
    if len(tokens) <= _MARKER_MAX_TOKENS and any(
        marker in tokens for marker in _FABRICATION_MARKERS
    ):
        return True
    return False


@functools.lru_cache(maxsize=1)
def _folded_phrases() -> dict[str, tuple[str, ...]]:
    """The blocklist, folded the same way segment text is.

    The phrases are written naturally (with hyphens and apostrophes), but they
    are compared against FOLDED text, so they have to be folded too — otherwise
    every entry containing one of those characters is dead weight.
    """
    return {
        lang: tuple(_fold(phrase).strip() for phrase in phrases)
        for lang, phrases in FABRICATION_PHRASES.items()
    }


def _tokens(text: str) -> list[str]:
    return _fold(text).split()


def looks_loopy(text: str, max_consecutive: int = 5, min_tokens: int = 8) -> bool:
    """Detect decoder repetition loops: a 1-3 token group repeated many times
    consecutively, or a long output with almost no distinct tokens."""
    toks = _tokens(text)
    if len(toks) < min_tokens:
        return False
    for size in (1, 2, 3):
        run = 1
        for i in range(size, len(toks) - size + 1, size):
            if toks[i : i + size] == toks[i - size : i]:
                run += 1
                if run > max_consecutive:
                    return True
            else:
                run = 1
    if len(toks) >= 12 and len(set(toks)) / len(toks) < 0.25:
        return True
    return False


def collapse_loops(text: str, keep: int = 3, max_consecutive: int = 5) -> str:
    """Last-resort textual cleanup: collapse >max_consecutive consecutive
    repeats of a short group down to `keep` occurrences. Works on the raw
    (unfolded) text so the surviving words keep their formatting. Genuine
    emphatic repetition ("no, no, no") stays; 14x loops shrink."""
    if not looks_loopy(text, max_consecutive=max_consecutive):
        return text
    words = text.split()
    for size in (3, 2, 1):
        out: list[str] = []
        i = 0
        while i < len(words):
            group = words[i : i + size]
            run = 1
            j = i + size
            while j + size <= len(words) + 1 and [_fold(w) for w in words[j : j + size]] == [
                _fold(w) for w in group
            ] and words[j : j + size]:
                run += 1
                j += size
            if run > max_consecutive:
                out.extend(words[i : i + size * keep])
                i = j
            else:
                out.append(words[i])
                i += 1
        words = out
    return " ".join(words)


def _token_overlap(a: str, b: str) -> float:
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def dedup_overlapping_segments(segments: list) -> list:
    """Drop near-duplicate transcriptions of overlapping time ranges.

    When two diarization turns overlap, both spans are transcribed from the
    same mono mix and the louder voice's words appear twice ("Buenas, ¿nos
    oyen?" twice within 2s). A segment is dropped when an earlier-started,
    LONGER segment overlaps it in time (>=60% of the shorter) and shares
    >=80% of its tokens. Distinct words in overlapping time (genuine
    interjections) always survive.

    Works on objects with start_seconds/end_seconds/text attributes; returns
    a new list preserving order.
    """
    kept: list = []
    for seg in sorted(segments, key=lambda s: ((s.start_seconds or 0), -((s.end_seconds or 0) - (s.start_seconds or 0)))):
        s0, s1 = seg.start_seconds or 0.0, seg.end_seconds or 0.0
        dur = max(0.0, s1 - s0)
        duplicate = False
        for other in kept:
            o0, o1 = other.start_seconds or 0.0, other.end_seconds or 0.0
            shared = max(0.0, min(s1, o1) - max(s0, o0))
            shorter = max(0.1, min(dur, max(0.0, o1 - o0)))
            if shared / shorter >= 0.6 and _token_overlap(seg.text or "", other.text or "") >= 0.8:
                # keep the longer rendition of the same words
                victim = seg if dur <= (o1 - o0) else other
                # An echo is a SUBSTANTIAL stretch of the same words. A one- or
                # two-word segment is a subset of any long text by base rate
                # (_token_overlap divides by min(), so a lone "Não" scores 1.0
                # against a 20-word turn containing "não"), and dropping it
                # deletes a real interjection by SOMEONE ELSE — field case: the
                # only thing a participant said in a meeting, a disagreement,
                # vanished because the other speaker used the word "não".
                # Below the floor we keep both: a visible duplicate costs less
                # than deleted speech.
                keeper = other if victim is seg else seg
                victim_tokens = _tokens(victim.text or "")
                # Short text is an echo only when it is the SAME text, not when
                # it is a subset of a longer turn: "Buenas, ¿nos oyen?" twice on
                # overlapping turns is one voice heard twice (drop it), while a
                # lone "Não" inside someone else's 20-word turn is a real
                # interjection (keep it).
                same_text = set(victim_tokens) == set(_tokens(keeper.text or ""))
                if len(victim_tokens) < _MIN_ECHO_TOKENS and not same_text:
                    continue
                if victim is seg:
                    duplicate = True
                    break
                kept.remove(other)
                break
        if not duplicate:
            kept.append(seg)
    kept.sort(key=lambda s: (s.start_seconds or 0))
    return kept
