"""Post-hoc batch speaker renaming for a processed meeting.

Why: with several people sharing one mic (hybrid rooms), Meet attributes all
in-room speech to the account owner, so the temporal vote names the first
in-room voice after the account and leaves the rest as SPEAKER_N. pyannote's
voice separation is still correct — only the labels are wrong, so a rename
sweep fixes the whole meeting at once.

Applies a {old_name -> new_name} mapping to the newest completed run:
transcript segments, the speaker_names map, action-item owners, AND name
mentions in the insights prose (summary/points/decisions/…). Prose names are
treated like variables: we know exactly which strings are participant names,
so they're substituted deterministically — full names always, bare first
names only when unambiguous among participants. latest.json is kept in sync
and every rename batch is
appended to speaker_renames.json in the run dir as an audit trail. Renames do
not survive a full reprocess (ASR + the vote regenerate labels from scratch).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_NAME_LENGTH = 80


def _clean(renames: dict) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for old, new in renames.items():
        old = (old or "").strip()
        new = (new or "").strip()
        if not old or not new or old == new:
            continue
        if len(new) > MAX_NAME_LENGTH:
            raise ValueError(f"Replacement name too long: {new[:40]}…")
        cleaned[old] = new
    return cleaned


_PROSE_LIST_FIELDS = ("most_important_points", "next_steps", "decisions", "risks_or_open_questions")


def _prose_substitutions(renames: dict[str, str], all_speakers: set[str]) -> list[tuple]:
    """Compiled (pattern, replacement) pairs for prose: full names always;
    bare first names only when unambiguous among participants (a lone 'Ana'
    in a meeting with two Anas stays untouched)."""
    import re

    first_name_owner: dict[str, list[str]] = {}
    for name in all_speakers | set(renames):
        first = name.split()[0] if name.split() else ""
        if first:
            first_name_owner.setdefault(first, []).append(name)

    subs: list[tuple] = []
    for old, new in renames.items():
        subs.append((re.compile(rf"(?<!\w){re.escape(old)}(?!\w)"), new))
        old_first = old.split()[0] if old.split() else ""
        new_first = new.split()[0] if new.split() else ""
        if (
            old_first
            and new_first
            and old_first != old  # multi-word name — bare-first-name mentions possible
            and len(first_name_owner.get(old_first, [])) == 1  # unambiguous
        ):
            subs.append((re.compile(rf"(?<!\w){re.escape(old_first)}(?!\w)"), new_first))
    return subs


def _sweep_prose(text: str, subs: list[tuple]) -> tuple[str, int]:
    count = 0
    for pattern, replacement in subs:
        text, n = pattern.subn(replacement, text)
        count += n
    return text, count


def _apply_to_result(result: dict, renames: dict[str, str]) -> tuple[int, int]:
    """Mutate a result.json-shaped dict in place.
    Returns (segments touched, prose mentions replaced)."""
    touched = 0
    transcript = result.get("transcript") or {}
    all_speakers = {
        s.get("speaker") for s in transcript.get("segments") or [] if s.get("speaker")
    } | set((transcript.get("speaker_names") or {}).values())
    for segment in transcript.get("segments") or []:
        speaker = segment.get("speaker")
        if speaker in renames:
            segment["speaker"] = renames[speaker]
            touched += 1
    names = transcript.get("speaker_names")
    if isinstance(names, dict):
        for label, name in list(names.items()):
            if name in renames:
                names[label] = renames[name]
            elif label in renames and label == name:
                names[label] = renames[label]
        # SPEAKER_N labels that were renamed directly get a mapping entry so
        # downstream consumers see them as resolved.
        for old, new in renames.items():
            if old.startswith("SPEAKER_") and old not in names:
                names[old] = new

    # Names behave like variables in the prose too: we know exactly which
    # strings are participant names, so sweep summary/points/decisions/etc.
    # deterministically (full names always, bare first names when unambiguous).
    subs = _prose_substitutions(renames, all_speakers)
    prose = 0
    insights = result.get("insights") or {}
    if isinstance(insights.get("summary"), str):
        insights["summary"], n = _sweep_prose(insights["summary"], subs)
        prose += n
    for field in _PROSE_LIST_FIELDS:
        items = insights.get(field)
        if isinstance(items, list):
            for i, item in enumerate(items):
                if isinstance(item, str):
                    items[i], n = _sweep_prose(item, subs)
                    prose += n
    for item in insights.get("action_items") or []:
        if not isinstance(item, dict):
            continue
        if item.get("owner") in renames:
            item["owner"] = renames[item["owner"]]
        for key in ("task", "notes"):
            if isinstance(item.get(key), str):
                item[key], n = _sweep_prose(item[key], subs)
                prose += n
    return touched, prose


def rename_speakers(meetings_dir: Path, meeting_id: str, renames: dict) -> dict:
    """Apply renames to the newest run with a result.json + latest.json.

    Returns {"renamed_segments": n, "run": run_name, "applied": {...}}.
    Raises FileNotFoundError when the meeting has no completed run,
    ValueError on unusable rename values.
    """
    renames = _clean(renames)
    if not renames:
        raise ValueError("No usable renames provided")

    meeting_dir = meetings_dir / str(meeting_id)
    run_dirs = sorted(
        (d for d in meeting_dir.iterdir() if d.is_dir() and (d / "result.json").is_file()),
        reverse=True,
    ) if meeting_dir.is_dir() else []
    if not run_dirs:
        raise FileNotFoundError(f"No completed run for meeting {meeting_id}")
    run_dir = run_dirs[0]

    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    touched, prose = _apply_to_result(result, renames)
    if touched == 0 and prose == 0 and not any(
        old in (result.get("transcript", {}).get("speaker_names") or {}).values()
        for old in renames
    ):
        raise ValueError("None of the given speakers exist in this meeting")

    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    # transcript.json mirrors the transcript part (used by LLM-only reprocess).
    transcript_path = run_dir / "transcript.json"
    if transcript_path.is_file():
        transcript_path.write_text(
            json.dumps(result.get("transcript") or {}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
    latest_path = meeting_dir / "latest.json"
    latest_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    audit_path = run_dir / "speaker_renames.json"
    try:
        history = json.loads(audit_path.read_text(encoding="utf-8"))
        if not isinstance(history, list):
            history = []
    except (OSError, json.JSONDecodeError):
        history = []
    history.append({"at": datetime.now(timezone.utc).isoformat(), "renames": renames})
    audit_path.write_text(json.dumps(history, ensure_ascii=False, indent=1), encoding="utf-8")

    logger.info(
        "Renamed speakers for meeting %s (%d segments, %d prose mentions) in %s: %s",
        meeting_id, touched, prose, run_dir.name, renames,
    )
    return {
        "renamed_segments": touched,
        "prose_replacements": prose,
        "run": run_dir.name,
        "applied": renames,
    }
