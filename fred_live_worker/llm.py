from __future__ import annotations

import asyncio
import difflib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .config import Settings

logger = logging.getLogger(__name__)
from .schemas import ActionItem, MeetingInsights, TranscriptResult, VexaMeeting
from .meeting_chat import transcript_for_llm


@dataclass
class LlmSummaryResult:
    insights: MeetingInsights
    raw_response: str
    raw_repair_response: str | None = None
    strategy: str = "single"
    raw_partial_responses: list[str] | None = None


SUMMARY_SCHEMA = {
    "summary": "string",
    "most_important_points": ["string"],
    "next_steps": ["string"],
    "decisions": ["string"],
    "action_items": [
        {
            "task": "string",
            "owner": "exactly one person's name, or null",
            "due_date": "string or null",
            "notes": "string or null",
        }
    ],
    "risks_or_open_questions": ["string"],
}


# The owner chip in the web UI renders ONE name with initials, so a compound
# value ("Acme (Jane Doe / John Roe)", "Jane Doe (John Roe)")
# renders as noise and conflates the speaker who said something with the person
# responsible for it. Field-observed with Qwen3.8 on a Catalan meeting.
_OWNER_RULE = (
    "The action-item owner must be exactly ONE person's name, written as it appears in the "
    "transcript: no organization, no role, no parentheses, no slashes, no alternative names. "
    "When a task belongs to a team rather than a named person, or the transcript does not "
    "clearly assign it to one person, use null instead of guessing. "
)


# Notes in two steps (LlamaCppClient._two_step). What a reader misses in notes
# written in one call is decisions, so a first pass lists them part by part and a
# second writes the notes from the whole transcript and that list.
_ITEM_DEFINITIONS = (
    "A decision is a choice the people in this meeting settled on. Status updates, facts and "
    "descriptions of how something works today are not decisions. An action item is a task a "
    "person took on or was asked to do, or work the group agreed on with nobody named yet "
    "(owner null). "
)
_OWNER_ATTRIBUTION = (
    "Whoever commits to a task owns it: after \"I'll send it\" the owner is the speaker of that "
    "line, and after a request to a named person who accepts, that person. "
)
# Asked to keep everything real, the writer did: 15 decisions per AMI meeting against
# the annotators' 5, and on real meetings a quarter of them repeated a task.
_MAX_DECISIONS = 8
_MAX_ACTION_ITEMS = 10
_LEAN_NOTES = (
    "Write for a colleague who missed the meeting and has one minute. In decisions keep only "
    "choices that change what the team does next, or that someone who was not there needs to "
    "know. In action_items keep only concrete tasks that matter. Leave out routine logistics, "
    f"small details and anything already said elsewhere in the notes. At most {_MAX_DECISIONS} "
    f"decisions and {_MAX_ACTION_ITEMS} action items, and fewer when the meeting produced fewer. "
    "Every fact appears once: a task goes in action_items and is not repeated in decisions, and "
    "that someone will do something is a task, not a decision. "
)
_EXTRACT_SCHEMA = {
    "decisions": SUMMARY_SCHEMA["decisions"],
    "action_items": SUMMARY_SCHEMA["action_items"],
}
# One list of things to do: asked for separately, next_steps restated the action
# items in other words, and text similarity cannot catch that.
_NOTES_SCHEMA = {key: value for key, value in SUMMARY_SCHEMA.items() if key != "next_steps"}
_EXTRACT_CHUNK_CHARS = 12000
# On top of max_tokens in the writing call, which counts the reasoning: AMI's half-hour
# meetings took 7-19k tokens, and one ran past 24k and left no answer.
_THINKING_TOKENS = 28000
MAX_GLOSSARY_CHARS = 3000
# The writing call carries the transcript AND the candidates; on real meetings the
# candidates ran to 15% of the transcript at most.
_TWO_STEP_HEADROOM = 1.3
# difflib ratio above which two task texts are the same task: the writer copies
# candidates almost word for word.
_SAME_TASK_RATIO = 0.6


def _owner_of(value: Any) -> str | None:
    """An owner, or None for the ways a model says there is none."""
    owner = str(value or "").strip()
    return None if owner.lower() in ("", "null", "none", "n/a") else owner


def _task_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _same_task(a: str, b: str) -> bool:
    return _task_similarity(a, b) >= _SAME_TASK_RATIO


def _restore_owners(items: list[ActionItem], candidates: list) -> None:
    """The writer copies the tasks and drops owners given as SPEAKER_N labels, asked
    or not (on a real meeting, 3 of 5 owners). A label is worth keeping: renaming the
    speaker on the meeting page renames the owner too."""
    owned = [
        (str(c.get("task") or ""), _owner_of(c.get("owner")))
        for c in candidates
        if isinstance(c, dict) and _owner_of(c.get("owner"))
    ]
    for item in items:
        item.owner = _owner_of(item.owner)
        if item.owner is None and owned:
            task, owner = max(owned, key=lambda pair: _task_similarity(item.task, pair[0]))
            if _same_task(item.task, task):
                item.owner = owner


def _reordered(items: list, order: Any) -> list:
    """`items` in the 1-based `order` a model returned; whatever it left out or
    repeated, each item stays exactly once and the unlisted ones keep their place
    at the end."""
    picked: list[int] = []
    for number in order if isinstance(order, list) else []:
        try:
            index = int(number) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(items) and index not in picked:
            picked.append(index)
    picked += [index for index in range(len(items)) if index not in picked]
    return [items[index] for index in picked]


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("LLM returned an empty response")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        raise ValueError("Could not find a JSON object in the LLM response")

    depth = 0
    in_string = False
    escape = False
    end = None

    for index, char in enumerate(text[start:], start=start):
        if escape:
            escape = False
            continue

        if char == "\\":
            escape = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break

    if end is None:
        raise ValueError("Could not find the end of the JSON object in the LLM response")

    return json.loads(text[start:end])


# Conservative for PT/CA/EN on SentencePiece vocabs (~3.2-4 chars/token); underestimating
# the budget just triggers map-reduce earlier, which is the safe direction.
_CHARS_PER_TOKEN = 3
_PROMPT_OVERHEAD_TOKENS = 1500  # system prompt + schema + meeting metadata + JSON wrapping
_MIN_TRANSCRIPT_BUDGET_TOKENS = 256


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _split_point(window: str) -> int:
    """Best split position in window, preferring paragraph > sentence > line > word breaks."""
    for sep in ("\n\n", ". ", "\n", " "):
        idx = window.rfind(sep)
        if idx >= len(window) // 2:
            return idx + len(sep)
    return len(window)


def _chunk_text(text: str, max_chars: int) -> list[str]:
    max_chars = max(1, max_chars)
    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        cut = _split_point(remaining[:max_chars])
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:]
    if remaining.strip():
        chunks.append(remaining.strip())
    return [chunk for chunk in chunks if chunk]


_STRING_LIST_FIELDS = ("most_important_points", "next_steps", "decisions", "risks_or_open_questions")


def _coerce_string_lists(data: dict[str, Any]) -> dict[str, Any]:
    """Coerce fields that should be lists of strings in case the model returns dicts."""
    for field in _STRING_LIST_FIELDS:
        items = data.get(field)
        if not isinstance(items, list):
            continue
        coerced = []
        for item in items:
            if isinstance(item, str):
                coerced.append(item)
            elif isinstance(item, dict):
                coerced.append(item.get("task") or next(iter(item.values()), str(item)))
            else:
                coerced.append(str(item))
        data[field] = coerced
    return data


# A remote endpoint (for example an Open WebUI on the organization's network)
# can blip: a VPN hiccup, a restart, a proxy's 502 while the backend reloads, or
# Open WebUI's 400 "Model not found", answered once on 2026-11-03 between calls
# that worked. Those are retried after a pause (a model name that is really
# wrong fails after the last one). Anything else (401 wrong key, 400 prompt too
# long, a read timeout after llm_timeout_seconds) fails at once, as before.
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_RETRY_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadError, httpx.RemoteProtocolError)
_SUMMARY_RETRY_DELAYS = (5.0, 15.0, 45.0)
# Ask is interactive (the site's proxy gives up after 180 s): one quick retry.
_ASK_RETRY_DELAYS = (2.0,)
_HTTPStatusError = httpx.HTTPStatusError
_HTTPError = httpx.HTTPError
_sleep = asyncio.sleep


def _retryable(response: httpx.Response) -> bool:
    return response.status_code in _RETRY_STATUS or (
        response.is_client_error and "model not found" in response.text.lower()
    )


def glossary_instruction(path: Path) -> str:
    """The prompt sentence for the team's own names, which speech recognition misspells
    ("Cuen" for Qwen, "Ember House" for ML House) and notes would copy: on 3 real meetings
    9 wrong forms in the notes became none, and the right names went from 5 to 22, with
    the same 61 decisions. Told only to write the name as listed, the writer once kept
    both ("Jev (Jeff)", twice in one meeting's notes), hence "alone". One name per line
    in the file, `term | what it is | how it comes out wrong`, with # for comments; a
    deployment's own list, and read at every call, so editing it needs no restart."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError):
        return ""
    names = []
    for line in lines:
        if line.lstrip().startswith("#"):
            continue
        term, what, wrong = ([field.strip() for field in line.split("|")] + ["", ""])[:3]
        if not term:
            continue
        details = "; ".join(filter(None, [what, f"it can come out as {wrong}" if wrong else ""]))
        names.append(f"{term} ({details})" if details else term)
    if not names:
        return ""
    listing = ", ".join(names)[:MAX_GLOSSARY_CHARS]
    return (
        "Speech recognition misspells this team's names. When the transcript has a "
        "similar-sounding word, write the name as listed here, alone, without the "
        f"transcript's spelling in parentheses: {listing}. "
    )


class LlamaCppClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def glossary(self) -> str:
        return glossary_instruction(self.settings.glossary_path)

    def _chat_completions_url(self) -> str:
        base = self.settings.llm_base_url.rstrip("/")
        # .../v1: llama.cpp, vLLM. .../api: Open WebUI (e.g. https://llm.example.org/api),
        # whose OpenAI-compatible route is /api/chat/completions.
        if base.endswith(("/v1", "/api")):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    async def _create_chat_completion(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        retry_delays: tuple[float, ...] = _SUMMARY_RETRY_DELAYS,
        thinking: bool = False,
    ) -> str:
        payload = {
            "model": self.settings.llm_model,
            "temperature": self.settings.llm_temperature if temperature is None else temperature,
            "max_tokens": self.settings.llm_max_tokens if max_tokens is None else max_tokens,
            "messages": messages,
        }
        if thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": True}
        elif self.settings.llm_disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        url = self._chat_completions_url()
        delays = iter(retry_delays)
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await client.post(url, headers=headers, json=payload)
            except _RETRY_ERRORS as exc:
                delay = next(delays, None)
                if delay is None:
                    raise
                reason = f"{type(exc).__name__}: {exc}"
            else:
                delay = next(delays, None) if _retryable(response) else None
                if delay is None:
                    break
                reason = f"HTTP {response.status_code}: {response.text[:100]}"
            logger.warning(
                "LLM request failed (%s); retry %d/%d in %.0fs",
                reason, attempt, len(retry_delays), delay,
            )
            await _sleep(delay)
        if response.is_error:
            # The body says why (Open WebUI: {"detail": "Model not found"}); httpx's own
            # message names only the status, and this one is what error.json shows.
            raise _HTTPStatusError(
                f"LLM endpoint answered {response.status_code}: {response.text[:300]}",
                request=response.request,
                response=response,
            )
        body = response.json()
        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(f"LLM response has no choices: {str(body)[:300]}") from None
        return message.get("content") or ""

    # 'same-as-transcript' must resolve to a CONCRETE language name: a vague
    # "same language as the transcript" sentence loses to the English prior of
    # the surrounding prompt+schema (observed: es/ca meeting summarized in
    # English by two different models). The worker already knows the dominant
    # language — say it by name, forcefully.
    _LANGUAGE_NAMES = {
        "en": "English", "es": "Spanish", "pt": "Portuguese", "pt-br": "Brazilian Portuguese",
        "ca": "Catalan",
        "fr": "French", "de": "German", "it": "Italian", "nl": "Dutch",
        "tr": "Turkish", "ru": "Russian", "pl": "Polish", "uk": "Ukrainian",
        "ar": "Arabic", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
        "hi": "Hindi", "sv": "Swedish", "da": "Danish", "no": "Norwegian",
        "fi": "Finnish", "cs": "Czech", "ro": "Romanian", "el": "Greek",
    }

    def _language_instruction(
        self, transcript_language: str | None = None, override: str | None = None
    ) -> str:
        if override:
            name = self._LANGUAGE_NAMES.get(override.lower(), override)
            return (
                f"IMPORTANT: write every JSON string value in {name}, regardless of the "
                "transcript's language (JSON keys stay as in the schema)."
            )
        target = self.settings.llm_output_language
        if target == "same-as-transcript":
            if not transcript_language:
                return "Write the JSON values in the same language as the transcript."
            name = self._LANGUAGE_NAMES.get(transcript_language.lower(), transcript_language)
            return (
                f"IMPORTANT: the meeting was held in {name}, so write every JSON string value "
                f"in {name}. Do not translate the content into English (JSON keys stay as in the schema)."
            )
        return (
            f"IMPORTANT: write every JSON string value in {target}, regardless of the "
            "transcript's language (JSON keys stay as in the schema)."
        )

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.llm_api_key.get_secret_value()}"
        return headers

    def _transcript_token_budget(self) -> int:
        budget = self.settings.llm_ctx_size - self.settings.llm_max_tokens - _PROMPT_OVERHEAD_TOKENS
        return max(_MIN_TRANSCRIPT_BUDGET_TOKENS, budget)

    async def _json_completion(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        messages: list[dict[str, str]],
    ) -> tuple[dict[str, Any], str, str | None]:
        """One completion parsed as JSON, with a temperature-0 repair retry on parse failure."""
        content = await self._create_chat_completion(client, headers, messages)
        logger.info("LLM raw response (first 500 chars): %s", content[:500])
        try:
            return _extract_json_object(content), content, None
        except ValueError:
            repair_prompt = (
                "Rewrite the following meeting analysis as valid JSON only. "
                "Do not add markdown, explanations, or any text before or after the JSON. "
                "If a value is missing in the source text, use null or an empty list as appropriate. "
                f"Use exactly this schema: {json.dumps(SUMMARY_SCHEMA, ensure_ascii=False)}\n\n"
                f"Source text:\n{content}"
            )
            raw_repair = await self._create_chat_completion(
                client,
                headers,
                [
                    {
                        "role": "system",
                        "content": "You convert meeting analyses into strict JSON objects.",
                    },
                    {"role": "user", "content": repair_prompt},
                ],
                temperature=0.0,
                max_tokens=self.settings.llm_max_tokens,
            )
            return _extract_json_object(raw_repair), content, raw_repair

    @staticmethod
    def _meeting_context(meeting: VexaMeeting) -> dict[str, Any]:
        return {
            "id": meeting.id,
            "platform": meeting.platform,
            "status": meeting.status,
            "duration_seconds": meeting.duration_seconds,
            "start_time": meeting.start_time,
            "end_time": meeting.end_time,
        }

    @staticmethod
    def _speaker_instruction(has_speakers: bool) -> str:
        if not has_speakers:
            return ""
        return (
            "Transcript lines are prefixed with the speaker (a real name or a SPEAKER_N label). "
            "Use them to attribute decisions and action-item owners. For SPEAKER_N labels, use a "
            "real name only when the conversation itself reveals it; never invent names. "
        )

    def _single_messages(
        self, meeting: VexaMeeting, transcript_text: str, has_speakers: bool = False,
        transcript_language: str | None = None, output_language: str | None = None,
    ) -> list[dict[str, str]]:
        system_prompt = (
            "You are a precise meeting analyst. "
            "Return valid JSON only and do not wrap it in markdown. "
            "Do not invent facts, owners, or dates. Use null when the transcript does not specify them. "
            f"{_OWNER_RULE}"
            f"{self._speaker_instruction(has_speakers)}{self.glossary()}"
            f"{self._language_instruction(transcript_language, output_language)} "
            f"Use this JSON schema: {json.dumps(SUMMARY_SCHEMA, ensure_ascii=False)}"
        )
        user_prompt = json.dumps(
            {"meeting": self._meeting_context(meeting), "transcript": transcript_text},
            ensure_ascii=False,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _chunk_messages(
        self, index: int, total: int, chunk: str, has_speakers: bool = False,
        transcript_language: str | None = None, output_language: str | None = None,
    ) -> list[dict[str, str]]:
        system_prompt = (
            f"You are a precise meeting analyst. You are given part {index} of {total} of one "
            "longer meeting transcript. "
            "Return valid JSON only and do not wrap it in markdown. "
            "Only include information stated in this part; do not speculate about other parts. "
            "Do not invent facts, owners, or dates. Use null when the transcript does not specify them. "
            f"{_OWNER_RULE}"
            f"{self._speaker_instruction(has_speakers)}{self.glossary()}"
            f"{self._language_instruction(transcript_language, output_language)} "
            f"Use this JSON schema: {json.dumps(SUMMARY_SCHEMA, ensure_ascii=False)}"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk},
        ]

    def _merge_messages(
        self, meeting: VexaMeeting, partials: list[dict[str, Any]],
        transcript_language: str | None = None, output_language: str | None = None,
    ) -> list[dict[str, str]]:
        system_prompt = (
            "You are a precise meeting analyst. You are given JSON analyses of consecutive parts "
            "of ONE meeting, in order. Merge them into a single JSON object with the same schema: "
            "write one coherent summary covering the whole meeting, deduplicate overlapping points, "
            "decisions and risks, and keep every distinct action item. "
            "Return valid JSON only and do not wrap it in markdown. "
            "Do not invent facts, owners, or dates. "
            f"{_OWNER_RULE}{self.glossary()}"
            f"{self._language_instruction(transcript_language, output_language)} "
            f"Use this JSON schema: {json.dumps(SUMMARY_SCHEMA, ensure_ascii=False)}"
        )
        user_prompt = json.dumps(
            {"meeting": self._meeting_context(meeting), "partial_analyses": partials},
            ensure_ascii=False,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _extract_messages(
        self, index: int, total: int, chunk: str, has_speakers: bool = False,
        transcript_language: str | None = None, output_language: str | None = None,
    ) -> list[dict[str, str]]:
        system_prompt = (
            f"You are a meticulous meeting analyst. You are given part {index} of {total} of one "
            "meeting transcript. List the decisions and commitments made in this part, and the "
            "tasks someone took on or was given. "
            "Do not list ideas that were only floated, and do not invent anything. "
            f"{_ITEM_DEFINITIONS}"
            "Return valid JSON only and do not wrap it in markdown. "
            f"{_OWNER_RULE}{_OWNER_ATTRIBUTION}"
            f"{self._speaker_instruction(has_speakers)}{self.glossary()}"
            f"{self._language_instruction(transcript_language, output_language)} "
            f"Use this JSON schema: {json.dumps(_EXTRACT_SCHEMA, ensure_ascii=False)}"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk},
        ]

    def _write_messages(
        self, meeting: VexaMeeting, transcript_text: str, candidates: dict[str, list],
        has_speakers: bool = False, transcript_language: str | None = None,
        output_language: str | None = None,
    ) -> list[dict[str, str]]:
        system_prompt = (
            "You are a precise meeting analyst. You are given the transcript of one meeting and "
            "candidate decisions and action items extracted from it part by part. Write the meeting "
            "notes. When the meeting later changed something, keep only the final state. "
            "Do not add facts that are not in the transcript. "
            "Use null when the transcript does not specify an owner or a date. "
            f"{_ITEM_DEFINITIONS}{_LEAN_NOTES}"
            "Keep the owner whenever a candidate or the transcript names one. "
            "Return valid JSON only and do not wrap it in markdown. "
            f"{_OWNER_RULE}{_OWNER_ATTRIBUTION}"
            f"{self._speaker_instruction(has_speakers)}{self.glossary()}"
            f"{self._language_instruction(transcript_language, output_language)} "
            f"Use this JSON schema: {json.dumps(_NOTES_SCHEMA, ensure_ascii=False)}"
        )
        user_prompt = json.dumps(
            {
                "meeting": self._meeting_context(meeting),
                "transcript": transcript_text,
                "candidates": candidates,
            },
            ensure_ascii=False,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def _write_notes(
        self, client: httpx.AsyncClient, headers: dict[str, str], messages: list[dict[str, str]],
    ) -> tuple[dict[str, Any], str, str | None]:
        """The writing call, reasoning first when llm_notes_thinking is on. The reasoning can
        run away and leave no answer, and an empty answer gives the JSON repair nothing to
        repair, so anything short of JSON gets the same call without it."""
        if self.settings.llm_notes_thinking:
            try:
                content = await self._create_chat_completion(
                    client, headers, messages,
                    max_tokens=self.settings.llm_max_tokens + _THINKING_TOKENS, thinking=True,
                )
                return _extract_json_object(content), content, None
            except (ValueError, RuntimeError, _HTTPError) as exc:
                logger.warning("Notes with reasoning failed (%s); writing them without it", exc)
        return await self._json_completion(client, headers, messages)

    async def _two_step(
        self, client: httpx.AsyncClient, headers: dict[str, str], meeting: VexaMeeting,
        transcript_text: str, has_speakers: bool, transcript_language: str | None,
        output_language: str | None,
    ) -> LlmSummaryResult:
        """Candidates part by part, then the notes from the whole transcript and the
        candidates. Against one call on AMI's 12 annotated meetings, with about as many
        items: decisions found 58-60% to 68-75%, items backed by the transcript 91-92% to
        93-95%. The steps after the writer are code, because the writer ignores
        instructions on them."""
        chunks = _chunk_text(transcript_text, _EXTRACT_CHUNK_CHARS)
        candidates: dict[str, list] = {"decisions": [], "action_items": []}
        partial_raws: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            try:
                parsed, raw, repair = await self._json_completion(
                    client, headers,
                    self._extract_messages(index, len(chunks), chunk, has_speakers, transcript_language, output_language),
                )
            except ValueError as exc:
                # The writing call still reads this part of the transcript.
                logger.warning("Extraction of part %d/%d unusable: %s", index, len(chunks), exc)
                continue
            parsed = _coerce_string_lists(parsed)
            candidates["decisions"] += parsed.get("decisions") or []
            candidates["action_items"] += parsed.get("action_items") or []
            partial_raws.append(repair or raw)
        logger.info(
            "Two-step notes: %d decisions and %d action items proposed from %d part(s)",
            len(candidates["decisions"]), len(candidates["action_items"]), len(chunks),
        )
        parsed, raw, raw_repair = await self._write_notes(
            client, headers,
            self._write_messages(meeting, transcript_text, candidates, has_speakers, transcript_language, output_language),
        )
        insights = MeetingInsights.model_validate(_coerce_string_lists(parsed))
        _restore_owners(insights.action_items, candidates["action_items"])
        ranking = await self._rank(client, headers, insights)
        if ranking:
            partial_raws.append(ranking)
        # Commitments first, then the work nobody took on yet; stable, so each
        # group keeps its order of consequence.
        insights.action_items.sort(key=lambda item: item.owner is None)
        if ranking:
            # The prompt sets these limits and the writer does not always keep them.
            del insights.decisions[_MAX_DECISIONS:]
            del insights.action_items[_MAX_ACTION_ITEMS:]
        return LlmSummaryResult(
            insights=insights,
            raw_response=raw,
            raw_repair_response=raw_repair,
            strategy="two-step",
            raw_partial_responses=partial_raws,
        )

    async def _rank(
        self, client: httpx.AsyncClient, headers: dict[str, str], insights: MeetingInsights
    ) -> str | None:
        """Most consequential first, in an order of its own: the writer keeps the
        candidates' order, however it is asked.
        Only the order is at stake here, so a failure leaves the notes as they are."""
        if len(insights.decisions) < 2 and len(insights.action_items) < 2:
            return None
        listing = {
            "summary": insights.summary,
            "decisions": {str(i): text for i, text in enumerate(insights.decisions, start=1)},
            "action_items": {str(i): item.task for i, item in enumerate(insights.action_items, start=1)},
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are given the numbered decisions and action items of one meeting. Order each "
                    "list by consequence for the people in the meeting: what changes direction, money, "
                    "deadlines or who is responsible comes first, operational detail last. Return valid "
                    'JSON only: {"decisions": [numbers], "action_items": [numbers]}, every number once.'
                ),
            },
            {"role": "user", "content": json.dumps(listing, ensure_ascii=False)},
        ]
        try:
            content = await self._create_chat_completion(client, headers, messages, temperature=0.0)
            order = _extract_json_object(content)
        except (ValueError, RuntimeError, _HTTPError) as exc:
            logger.warning("Ranking the notes failed (%s); they stay in the writer's order", exc)
            return None
        insights.decisions = _reordered(insights.decisions, order.get("decisions"))
        insights.action_items = _reordered(insights.action_items, order.get("action_items"))
        return content

    async def chat(self, messages: list[dict[str, str]], max_tokens: int | None = None) -> str:
        """One free-form chat completion (used by the ask endpoint)."""
        async with httpx.AsyncClient(timeout=self.settings.llm_timeout_seconds) as client:
            return await self._create_chat_completion(
                client, self._headers(), messages, max_tokens=max_tokens,
                retry_delays=_ASK_RETRY_DELAYS,
            )

    async def summarize(
        self, meeting: VexaMeeting, transcript: TranscriptResult,
        output_language: str | None = None,
    ) -> LlmSummaryResult:
        has_speakers = any(segment.speaker for segment in transcript.segments)
        # Speech, then the meeting chat as a labelled block (links and numbers
        # pasted there can be quoted in the notes, never mistaken for speech).
        transcript_text = transcript_for_llm(transcript).strip()
        if not transcript_text:
            raise ValueError("Cannot summarize an empty transcript")

        headers = self._headers()
        budget_tokens = self._transcript_token_budget()

        async with httpx.AsyncClient(timeout=self.settings.llm_timeout_seconds) as client:
            if (
                self.settings.llm_two_step
                and _estimate_tokens(transcript_text) * _TWO_STEP_HEADROOM <= budget_tokens
            ):
                try:
                    return await self._two_step(
                        client, headers, meeting, transcript_text, has_speakers,
                        transcript.language, output_language,
                    )
                except ValueError as exc:
                    # Long notes can hit max_tokens and come back as broken JSON.
                    # One call writes less: a meeting never loses its notes to this.
                    logger.warning("Two-step notes failed (%s); writing them in one call", exc)
            if _estimate_tokens(transcript_text) <= budget_tokens:
                parsed, raw, raw_repair = await self._json_completion(
                    client, headers, self._single_messages(meeting, transcript_text, has_speakers, transcript.language, output_language)
                )
                strategy = "single"
                partial_raws: list[str] | None = None
            else:
                strategy = "map-reduce"
                chunk_chars = max(1, budget_tokens * _CHARS_PER_TOKEN * 6 // 10)
                chunks = _chunk_text(transcript_text, chunk_chars)
                logger.info(
                    "Transcript ~%d tokens exceeds budget %d; map-reduce over %d chunks",
                    _estimate_tokens(transcript_text),
                    budget_tokens,
                    len(chunks),
                )
                partials: list[dict[str, Any]] = []
                partial_raws = []
                for index, chunk in enumerate(chunks, start=1):
                    parsed_chunk, raw_chunk, repair_chunk = await self._json_completion(
                        client, headers, self._chunk_messages(index, len(chunks), chunk, has_speakers, transcript.language, output_language)
                    )
                    partials.append(_coerce_string_lists(parsed_chunk))
                    partial_raws.append(repair_chunk or raw_chunk)
                parsed, raw, raw_repair = await self._json_completion(
                    client, headers, self._merge_messages(meeting, partials, transcript.language, output_language)
                )

        parsed = _coerce_string_lists(parsed)
        insights = MeetingInsights.model_validate(parsed)
        return LlmSummaryResult(
            insights=insights,
            raw_response=raw,
            raw_repair_response=raw_repair,
            strategy=strategy,
            raw_partial_responses=partial_raws,
        )
