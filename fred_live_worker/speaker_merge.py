"""Pure merge logic for speaker attribution.

Combines three time-aligned inputs, all relative to recording audio t=0:
- Whisper word timestamps (what was said, when)
- pyannote diarization turns (which anonymous speaker was talking, when)
- bot speaker events from the hook (which named participant was talking, when)

No GPU or network dependencies; fully unit-testable.
"""
from __future__ import annotations

import bisect
import logging
import re
import unicodedata
from dataclasses import dataclass

from .schemas import TranscriptSegment

logger = logging.getLogger(__name__)


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str  # diarization label, e.g. "SPEAKER_00"


@dataclass
class NamedInterval:
    start: float
    end: float
    name: str


def words_to_speaker_segments(words: list[Word], turns: list[SpeakerTurn]) -> list[TranscriptSegment]:
    """Assign each word to the diarization turn containing its midpoint, then group
    consecutive same-speaker words into segments. Words outside every turn inherit
    the previous word's speaker (None at the start)."""
    if not words:
        return []

    sorted_turns = sorted(turns, key=lambda t: t.start)

    def speaker_at(instant: float) -> str | None:
        for turn in sorted_turns:
            if turn.start <= instant <= turn.end:
                return turn.speaker
            if turn.start > instant:
                break
        return None

    segments: list[TranscriptSegment] = []
    current_words: list[Word] = []
    current_speaker: str | None = None

    def flush() -> None:
        if not current_words:
            return
        segments.append(
            TranscriptSegment(
                start_seconds=current_words[0].start,
                end_seconds=current_words[-1].end,
                text=" ".join(w.text.strip() for w in current_words if w.text.strip()),
                speaker=current_speaker,
            )
        )

    previous_speaker: str | None = None
    for word in words:
        midpoint = (word.start + word.end) / 2
        speaker = speaker_at(midpoint)
        if speaker is None:
            speaker = previous_speaker
        if speaker != current_speaker and current_words:
            flush()
            current_words = []
        current_speaker = speaker
        current_words.append(word)
        previous_speaker = speaker
    flush()

    return segments


_PRESENTATION_MARKERS = (
    "sua apresentação",
    "your presentation",
    "(presentation)",
    "apresentação de tela",
)


def _is_nonspeaker_name(name: str, participant_id: str) -> bool:
    """True when this event's name is not a person.

    Two structural tells, both observed in the field and both independent of how
    LONG the identity spoke — a duration floor is NOT usable here, because real
    participants accumulate as little as 0.1-3.0s of DOM speaking signal (Zoom
    exposes one active speaker, Teams' indicator flickers), overlapping entirely
    with the junk:

    (1) the name is the participant id, or one of its path components. Meet ids
        look like "spaces/<space>/devices/<n>"; a tile whose name node cannot be
        read has yielded both the whole id and the bare component "devices" as a
        name, which then counts as a distinct identity and inflates the
        diarization floor (field: a 3-person meeting got min_speakers=4 and
        pyannote split one voice in two, leaving an unnamed SPEAKER_02).
    (2) screen-share tiles ("Ada Lovelace (sua apresentação)"), which carry a
        real person's name but are not a voice.
    """
    lowered = name.casefold()
    if any(marker in lowered for marker in _PRESENTATION_MARKERS):
        return True
    pid = participant_id.strip()
    if not pid or "/" not in pid:
        return False
    if name == pid:
        return True
    return name in {part for part in pid.split("/") if part}


def drop_nonspeaker_events(speaker_events: list[dict]) -> list[dict]:
    """Filter DOM-artifact events out of a hook's speaker_events.

    Applied once, upstream of everything that reads them, so the junk can neither
    inflate min_speakers nor enter the name vote / Teams assignment name pool."""
    kept: list[dict] = []
    dropped: dict[str, int] = {}
    for event in speaker_events:
        if not isinstance(event, dict):
            continue
        name = str(event.get("participant_name") or "").strip()
        pid = str(event.get("participant_id") or "")
        if name and _is_nonspeaker_name(name, pid):
            dropped[name] = dropped.get(name, 0) + 1
            continue
        kept.append(event)
    if dropped:
        logger.info(
            "Speaker events: dropped %d non-speaker event(s) %s",
            sum(dropped.values()),
            ", ".join(f"{n!r}x{c}" for n, c in sorted(dropped.items())),
        )
    return kept


def events_to_named_intervals(
    speaker_events: list[dict],
    audio_duration_seconds: float | None,
) -> list[NamedInterval]:
    """Pair SPEAKER_START/SPEAKER_END bot events into named speaking intervals.

    Bot-side caveats handled here: dangling STARTs (participant left mid-speech)
    are clipped to audio duration; events past the audio duration are dropped;
    unmatched ENDs are ignored."""
    open_starts: dict[str, tuple[str, float]] = {}  # participant_id -> (name, start_s)
    intervals: list[NamedInterval] = []

    for event in speaker_events:
        try:
            # upstream Vexa emits relative_timestamp_ms; our retired bot patch used
            # relative_ts_ms — accept both so saved hooks from either era parse.
            raw_ts = event.get("relative_timestamp_ms", event.get("relative_ts_ms"))
            ts = float(raw_ts) / 1000.0
            etype = event["event_type"]
            pid = str(event.get("participant_id") or event.get("participant_name") or "")
            name = str(event.get("participant_name") or "").strip()
        except (KeyError, TypeError, ValueError):
            continue
        if not name or ts < 0:
            continue
        if audio_duration_seconds is not None and ts > audio_duration_seconds:
            ts = audio_duration_seconds

        if etype == "SPEAKER_START":
            open_starts[pid] = (name, ts)
        elif etype == "SPEAKER_END":
            started = open_starts.pop(pid, None)
            if started is not None:
                name0, start_s = started
                if ts > start_s:
                    intervals.append(NamedInterval(start=start_s, end=ts, name=name0))

    if audio_duration_seconds is not None:
        for name0, start_s in open_starts.values():
            if audio_duration_seconds > start_s:
                intervals.append(NamedInterval(start=start_s, end=audio_duration_seconds, name=name0))

    return sorted(intervals, key=lambda iv: iv.start)


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def map_speakers_to_names(
    turns: list[SpeakerTurn],
    named_intervals: list[NamedInterval],
    min_overlap_fraction: float = 0.25,
    min_overlap_seconds: float = 3.0,
    dominance_ratio: float = 1.5,
    min_overlap_floor_seconds: float = 1.0,
    min_rival_share: float = 0.05,
) -> dict[str, str]:
    """Map each diarization label to the participant name whose speaking intervals
    overlap it the most. Meet's active-speaker indicator lags real speech (events
    typically cover only ~half of it), so instead of a strict majority we require:
    winner covers >= min_overlap_fraction of the label's speaking time, with
    enough absolute evidence, and beats the runner-up by dominance_ratio.
    Ambiguous labels stay unmapped.

    The absolute-evidence bar SCALES with the cluster: min(min_overlap_seconds,
    max(floor, 0.5 * cluster_duration)). A fixed bar is unattainable for small
    clusters by construction (field failure: a participant with 5.5s of turns
    had 58% event coverage yet failed the fixed 3.0s bar, staying SPEAKER_N).
    Quality is still enforced by the fraction and dominance gates; the floor
    keeps single-blip flukes out.

    Dominance compares LIFT (overlap vs what a candidate would hit by chance
    given their share of the meeting), not raw overlap: a dominant talker's
    events blanket the timeline, so on any small cluster they rack up raw
    overlap by base rate alone (same field failure: the 78%-of-meeting talker
    scored 2.29s on the quiet participant's 5.5s cluster purely by chance,
    inflating the runner-up past the 1.5x raw gate). A rival counts only with
    min_rival_share of the cluster: lift divides by a candidate's time in the
    whole meeting, so someone with one second of signal reaches any lift with one
    second of overlap (field failure: such a tile kept a 158s cluster, 85% lit by
    its real owner, unnamed). Nor is someone their own rival when they rejoined
    under another display name."""
    total_by_label: dict[str, float] = {}
    overlap_by_label_name: dict[str, dict[str, float]] = {}

    for turn in turns:
        duration = max(0.0, turn.end - turn.start)
        total_by_label[turn.speaker] = total_by_label.get(turn.speaker, 0.0) + duration
        by_name = overlap_by_label_name.setdefault(turn.speaker, {})
        for interval in named_intervals:
            shared = _overlap(turn.start, turn.end, interval.start, interval.end)
            if shared > 0:
                by_name[interval.name] = by_name.get(interval.name, 0.0) + shared

    # Each candidate's share of the meeting timeline, for chance-normalization.
    span_end = max(
        [turn.end for turn in turns] + [i.end for i in named_intervals], default=0.0
    )
    coverage_by_name: dict[str, float] = {}
    for interval in named_intervals:
        coverage_by_name[interval.name] = coverage_by_name.get(interval.name, 0.0) + max(
            0.0, interval.end - interval.start
        )

    def lift(name: str, overlap: float, cluster_seconds: float) -> float:
        if span_end <= 0:
            return 0.0
        expected = cluster_seconds * (coverage_by_name.get(name, 0.0) / span_end)
        return overlap / max(expected, 0.05)

    mapping: dict[str, str] = {}
    for label, by_name in overlap_by_label_name.items():
        if not by_name:
            continue
        ranked = sorted(by_name.items(), key=lambda item: item[1], reverse=True)
        best_name, best_overlap = ranked[0]
        total = total_by_label.get(label, 0.0)
        if total <= 0:
            continue
        required_seconds = min(
            min_overlap_seconds, max(min_overlap_floor_seconds, 0.5 * total)
        )
        best_lift = lift(best_name, best_overlap, total)
        rival_lift = max(
            (
                lift(name, ov, total)
                for name, ov in ranked[1:]
                if ov / total >= min_rival_share and not same_person(name, best_name)
            ),
            default=0.0,
        )
        accepted = (
            best_overlap / total >= min_overlap_fraction
            and best_overlap >= required_seconds
            and best_lift >= dominance_ratio * max(rival_lift, 1e-9)
        )
        logger.info(
            "Name vote %s: turns=%.1fs best=%s %.2fs (%.0f%%, lift %.1f) "
            "rival-lift=%.1f required>=%.2fs -> %s",
            label, total, best_name, best_overlap, 100 * best_overlap / total,
            best_lift, rival_lift, required_seconds,
            best_name if accepted else "unmapped",
        )
        if accepted:
            mapping[label] = best_name

    # Two labels must never map to the same name (diarization may oversplit a
    # speaker, but silently merging identities hides errors — keep the strongest).
    by_name_strength: dict[str, tuple[str, float]] = {}
    for label, name in mapping.items():
        strength = overlap_by_label_name[label][name]
        incumbent = by_name_strength.get(name)
        if incumbent is None or strength > incumbent[1]:
            by_name_strength[name] = (label, strength)
    return {label: name for name, (label, _) in by_name_strength.items()}


def assignment_fallback_names(
    turns: list[SpeakerTurn],
    named_intervals: list[NamedInterval],
    mapping: dict[str, str],
    platform: str | None,
    min_pair_seconds: float = 45.0,
    min_joint_margin: float = 1.25,
    max_pool: int = 7,
) -> dict[str, str]:
    """Joint-assignment naming for clusters the per-cluster vote refused.
    TEAMS ONLY: its speaking indicator covers ~13% of real speech and
    cross-fires during other people's turns (measured on a real meeting), so
    per-cluster votes starve — but names are an ASSIGNMENT with exclusivity,
    and the jointly best pairing is decidable even when each row is a coin
    flip (field case: a name individually preferred the WRONG cluster; the
    joint optimum named both correctly at 1.36x margin).

    Hybrid-room safety (the shared-mic case where DOM names lie
    systematically, not just noisily):
      * names already claimed by any cluster (DOM vote or voiceprint
        override) leave the pool — a room guest can never inherit the mic
        owner's name;
      * needs >=2 unclaimed names AND >=2 unnamed clusters — a single
        smeared identity has no exclusivity to exploit and never plays;
      * per-pair floor (min_pair_seconds) and a joint margin vs the
        second-best assignment — symmetric smear collapses the margin
        toward 1.0 and is refused.
    Accepted only as a WHOLE assignment of >=2 pairs."""
    from itertools import permutations

    if (platform or "").strip().lower() != "teams":
        return {}
    claimed_names = set(mapping.values())
    labels = sorted({t.speaker for t in turns} - set(mapping))
    names = sorted({i.name for i in named_intervals} - claimed_names)
    if len(labels) < 2 or len(names) < 2:
        return {}
    if len(labels) > max_pool or len(names) > max_pool:
        logger.info("Assignment fallback: pool too large (%d names x %d clusters); staying silent",
                    len(names), len(labels))
        return {}

    ov: dict[tuple[str, str], float] = {}
    for turn in turns:
        if turn.speaker not in labels:
            continue
        for iv in named_intervals:
            if iv.name not in names:
                continue
            shared = _overlap(turn.start, turn.end, iv.start, iv.end)
            if shared > 0:
                key = (iv.name, turn.speaker)
                ov[key] = ov.get(key, 0.0) + shared
    allowed = {k: v for k, v in ov.items() if v >= min_pair_seconds}
    if not allowed:
        return {}

    # Enumerate injective assignments over allowed cells (pools are tiny).
    if len(names) <= len(labels):
        rows, cols, row_is_name = names, labels, True
    else:
        rows, cols, row_is_name = labels, names, False
    scored: list[tuple[float, frozenset]] = []
    for perm in permutations(cols, len(rows)):
        pairs = []
        total = 0.0
        for row, col in zip(rows, perm):
            key = (row, col) if row_is_name else (col, row)
            if key in allowed:
                pairs.append(key)
                total += allowed[key]
        if len(pairs) >= 2:
            scored.append((total, frozenset(pairs)))
    if not scored:
        return {}
    scored.sort(key=lambda s: -s[0])
    best_total, best_pairs = scored[0]
    second_total = next((tot for tot, pairs in scored if pairs != best_pairs), None)
    margin = (best_total / second_total) if second_total else float("inf")
    result = {label: name for name, label in best_pairs}
    desc = ", ".join(f"{l}->{n}" for l, n in sorted(result.items()))
    if second_total is not None and margin < min_joint_margin:
        logger.info(
            "Assignment fallback (teams) refused: best %s joint=%.1fs vs %.1fs (%.2fx < %.2fx)",
            desc, best_total, second_total, margin, min_joint_margin,
        )
        return {}
    logger.info(
        "Assignment fallback (teams): %s joint=%.1fs vs %s (%.2fx)",
        desc, best_total,
        f"{second_total:.1f}s" if second_total else "no alternative",
        margin,
    )
    return result


def _name_words(name: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", name.casefold())
    return re.findall(r"[a-z0-9]+", "".join(c for c in folded if not unicodedata.combining(c)))


def same_person(a: str | None, b: str | None) -> bool:
    """Whether two display names can be one person: a tile says "Joaquin Sorolla" or
    "Alan T" where the print says "Joaquín Sorolla" and "Alan Turing". Every word of the
    shorter name must open a word of the longer one."""
    shorter, longer = sorted((_name_words(a or ""), _name_words(b or "")), key=len)
    return bool(shorter) and all(any(w.startswith(s) for w in longer) for s in shorter)


# A turn is a candidate when the tile was lit for this share of it. Meet's indicator
# lags and covers about half of real speech, so a turn 40% lit is usually mostly
# that person's.
PAGE_TURN_COVER = 0.4
# Under this a group is too little audio for a voice decision.
PAGE_GROUP_MIN_SECONDS = 3.0


@dataclass
class TurnGroup:
    """Turns of one cluster that may be one person's: their tile was lit (page_groups)
    or the voice is theirs (voiceprints.identify_turns)."""
    identity: str
    label: str
    turns: list[SpeakerTurn]

    @property
    def seconds(self) -> float:
        return sum(turn.end - turn.start for turn in self.turns)


def page_groups(
    turns: list[SpeakerTurn],
    named_intervals: list[NamedInterval],
    mapping: dict[str, str],
    min_cover: float = PAGE_TURN_COVER,
    min_seconds: float = PAGE_GROUP_MIN_SECONDS,
    min_turn_seconds: float = 0.3,
) -> list[TurnGroup]:
    """Where the page says someone spoke inside a cluster that is not theirs.

    The name vote and the voiceprints name whole clusters, so a person diarization
    folded into someone else's cluster has no way out: their name is simply missing.
    The page still knows when their tile was lit. These groups are only candidates:
    the indicator cross-fires, and on a room's device it lights up for every voice in
    the room, so the voice decides (voiceprints.py identify_page_groups)."""
    lit: dict[str, list[tuple[float, float]]] = {}
    for interval in named_intervals:
        lit.setdefault(interval.name, []).append((interval.start, interval.end))
    groups: list[TurnGroup] = []
    for name, spans in lit.items():
        starts, ends = _merged_spans(spans)
        own = {label for label, owner in mapping.items() if same_person(owner, name)}
        by_label: dict[str, list[SpeakerTurn]] = {}
        for turn in turns:
            length = turn.end - turn.start
            if length < min_turn_seconds or turn.speaker in own:
                continue
            if _seconds_inside(turn.start, turn.end, starts, ends) / length >= min_cover:
                by_label.setdefault(turn.speaker, []).append(turn)
        for label in sorted(by_label):
            group = TurnGroup(identity=name, label=label, turns=by_label[label])
            if group.seconds >= min_seconds:
                groups.append(group)
    return groups


def _merged_spans(spans: list[tuple[float, float]]) -> tuple[list[float], list[float]]:
    starts: list[float] = []
    ends: list[float] = []
    for start, end in sorted(spans):
        if ends and start <= ends[-1]:
            ends[-1] = max(ends[-1], end)
        else:
            starts.append(start)
            ends.append(end)
    return starts, ends


def _seconds_inside(start: float, end: float, starts: list[float], ends: list[float]) -> float:
    """Seconds of [start, end] inside merged spans, which a whole meeting's turns
    query thousands of times."""
    index = max(0, bisect.bisect_right(starts, start) - 1)
    total = 0.0
    while index < len(starts) and starts[index] < end:
        total += _overlap(start, end, starts[index], ends[index])
        index += 1
    return total


def move_turns(
    turns: list[SpeakerTurn],
    moves: list[tuple[TurnGroup, str]],
    mapping: dict[str, str],
) -> tuple[list[SpeakerTurn], dict[str, str]]:
    """Give each (group, name) its turns: to the name's longest cluster, or to a new
    label when nobody carries that name yet. A turn two groups claim goes to the
    first; the caller orders moves by how sure it is."""
    speech: dict[str, float] = {}
    for turn in turns:
        speech[turn.speaker] = speech.get(turn.speaker, 0.0) + turn.end - turn.start
    mapping = dict(mapping)
    destination: dict[tuple[float, float, str], str] = {}
    for group, name in moves:
        owned = [label for label, owner in mapping.items() if same_person(owner, name)]
        if owned:
            target = max(owned, key=lambda label: speech.get(label, 0.0))
        else:
            index = 0
            while f"SPEAKER_{index:02d}" in speech or f"SPEAKER_{index:02d}" in mapping:
                index += 1
            target = f"SPEAKER_{index:02d}"
            mapping[target] = name
        for turn in group.turns:
            destination.setdefault((turn.start, turn.end, turn.speaker), target)
    moved = [
        SpeakerTurn(t.start, t.end, destination.get((t.start, t.end, t.speaker), t.speaker))
        for t in turns
    ]
    return moved, mapping


def relabel_segments(
    segments: list[TranscriptSegment],
    mapping: dict[str, str],
) -> list[TranscriptSegment]:
    return [
        segment.model_copy(update={"speaker": mapping.get(segment.speaker, segment.speaker)})
        if segment.speaker is not None
        else segment
        for segment in segments
    ]


# Below this a turn is a short answer or a backchannel, and worth transcribing
# only when it stands alone: inside someone else's turn the mono mix hands
# Whisper the louder voice (AMI: 31% of the words real on fully overlapped short
# turns vs 63% on clean ones, NOTSOFAR 14% vs 36%).
def fuse_turns(
    nemotron: list[SpeakerTurn], pyannote: list[SpeakerTurn], min_gap: float = 0.2
) -> list[SpeakerTurn]:
    """Nemotron's turns, plus pyannote's speech where Nemotron heard nobody.

    The two fail in opposite ways: Nemotron separates voices better in a room
    (DER on NOTSOFAR 29.6 for pyannote, 17.3 for Nemotron) and misses more speech
    (AMI's room microphone, 20.0 against 26.8). Taking who from Nemotron and the
    speech it missed from pyannote beat both everywhere: 16.0 on NOTSOFAR, 17.6 on
    AMI, 24.0 on Bangor Miami. Each missed piece goes to the Nemotron speaker that
    pyannote's speaker overlaps most; a pyannote speaker Nemotron never heard stays
    a speaker of its own.
    """
    votes: dict[str, dict[str, float]] = {}
    for ours in pyannote:
        for theirs in nemotron:
            shared = _overlap(ours.start, ours.end, theirs.start, theirs.end)
            if shared > 0:
                tally = votes.setdefault(ours.speaker, {})
                tally[theirs.speaker] = tally.get(theirs.speaker, 0.0) + shared
    heirs = {speaker: max(tally, key=tally.get) for speaker, tally in votes.items()}

    taken = {turn.speaker for turn in nemotron}
    for speaker in sorted({turn.speaker for turn in pyannote} - set(heirs)):
        index = 0
        while f"SPEAKER_{index:02d}" in taken:
            index += 1
        heirs[speaker] = f"SPEAKER_{index:02d}"
        taken.add(heirs[speaker])

    covered: list[list[float]] = []
    for turn in sorted(nemotron, key=lambda t: t.start):
        if covered and turn.start <= covered[-1][1]:
            covered[-1][1] = max(covered[-1][1], turn.end)
        else:
            covered.append([turn.start, turn.end])

    fused = list(nemotron)
    for turn in pyannote:
        position = turn.start
        for start, end in covered:
            if end <= position:
                continue
            if start >= turn.end:
                break
            if start - position >= min_gap:
                fused.append(SpeakerTurn(position, start, heirs[turn.speaker]))
            position = end
        if turn.end - position >= min_gap:
            fused.append(SpeakerTurn(position, turn.end, heirs[turn.speaker]))
    return sorted(fused, key=lambda t: t.start)


SHORT_TURN_SECONDS = 1.0
SHORT_TURN_MAX_OVERLAP = 0.5


@dataclass
class TurnBlock:
    """A contiguous same-speaker stretch, the transcription unit for multilingual
    ASR: each block gets its own language detection."""
    start: float
    end: float
    speaker: str


def _share_covered(start: float, end: float, spans: list[tuple[float, float]]) -> float:
    """Share of [start, end] inside the union of spans."""
    covered = 0.0
    reached = start
    for span_start, span_end in sorted(spans):
        span_start, span_end = max(span_start, reached), min(span_end, end)
        if span_end > span_start:
            covered += span_end - span_start
            reached = span_end
    return covered / (end - start)


def group_turns_into_blocks(
    turns: list[SpeakerTurn],
    max_gap_seconds: float = 1.5,
    max_block_seconds: float = 60.0,
    min_block_seconds: float = 0.3,
) -> list[TurnBlock]:
    """Merge consecutive turns of the same speaker (bridging gaps up to
    max_gap_seconds) into blocks, splitting at max_block_seconds. Blocks shorter
    than min_block_seconds are dropped (pyannote flicker, not speech), and so are
    blocks under SHORT_TURN_SECONDS that other speakers' turns mostly cover."""
    blocks: list[TurnBlock] = []
    for turn in sorted(turns, key=lambda t: t.start):
        if (
            blocks
            and blocks[-1].speaker == turn.speaker
            and turn.start - blocks[-1].end <= max_gap_seconds
            and turn.end - blocks[-1].start <= max_block_seconds
        ):
            blocks[-1].end = max(blocks[-1].end, turn.end)
        else:
            blocks.append(TurnBlock(start=turn.start, end=turn.end, speaker=turn.speaker))
    kept: list[TurnBlock] = []
    flicker: list[TurnBlock] = []
    buried: list[TurnBlock] = []
    for block in blocks:
        length = block.end - block.start
        if length < min_block_seconds:
            flicker.append(block)
        elif length < SHORT_TURN_SECONDS and _share_covered(
            block.start, block.end,
            [(t.start, t.end) for t in turns if t.speaker != block.speaker],
        ) >= SHORT_TURN_MAX_OVERLAP:
            buried.append(block)
        else:
            kept.append(block)
    # Logged so the loss is auditable: nobody transcribes these windows.
    if flicker:
        logger.info(
            "Blocks: dropped %d turn(s) under %.1fs: %s",
            len(flicker),
            min_block_seconds,
            ", ".join(f"{b.speaker} {b.start:.1f}-{b.end:.1f}s" for b in flicker[:8]),
        )
    if buried:
        logger.info(
            "Blocks: dropped %d short turn(s) inside other speakers' turns: %s",
            len(buried),
            ", ".join(f"{b.speaker} {b.start:.1f}-{b.end:.1f}s" for b in buried[:8]),
        )
    return kept


def render_speaker_transcript(segments: list[TranscriptSegment]) -> str:
    """Render 'Name: text' lines for the LLM, merging consecutive same-speaker segments."""
    lines: list[str] = []
    current_speaker: str | None = None
    current_texts: list[str] = []

    def flush() -> None:
        if not current_texts:
            return
        prefix = f"{current_speaker}: " if current_speaker else ""
        lines.append(prefix + " ".join(current_texts))

    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        if segment.speaker != current_speaker and current_texts:
            flush()
            current_texts = []
        current_speaker = segment.speaker
        current_texts.append(text)
    flush()

    return "\n".join(lines)


def find_pause_splits(
    waveform,
    block_start: float,
    block_end: float,
    sample_rate: int = 16000,
    min_pause: float = 0.6,
    min_chunk: float = 3.0,
    rms_threshold: float = 0.01,
) -> list[float]:
    """Split points (seconds) at silent gaps inside one speaker block.

    Frames the block at 100ms, finds silent runs >= min_pause, and splits at
    each run's center — but never creates a chunk shorter than min_chunk.
    These are the candidate language-switch boundaries for intra-block LID:
    people code-switch at breaths, not mid-word.
    """
    import numpy as np

    frame = int(sample_rate * 0.1)
    lo = max(0, int(block_start * sample_rate))
    hi = min(len(waveform), int(block_end * sample_rate))
    if hi - lo < frame:
        return []
    section = waveform[lo : hi - ((hi - lo) % frame)]
    if section.size == 0:
        return []
    frames = section.reshape(-1, frame)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    silent = rms < rms_threshold

    splits: list[float] = []
    last_boundary = block_start
    run_start = None
    for i, is_silent in enumerate(silent):
        if is_silent and run_start is None:
            run_start = i
        elif not is_silent and run_start is not None:
            run_len = (i - run_start) * 0.1
            if run_len >= min_pause:
                candidate = block_start + (run_start + (i - run_start) / 2) * 0.1
                if candidate - last_boundary >= min_chunk and block_end - candidate >= min_chunk:
                    splits.append(round(candidate, 3))
                    last_boundary = candidate
            run_start = None
    return splits


def merge_language_spans(
    spans: list[tuple[float, float, str | None]],
) -> list[tuple[float, float, str | None]]:
    """Coalesce consecutive (start, end, language) spans that share a language.
    Unknown languages (None, e.g. low-confidence LID) inherit the previous
    span's language — or the next one's for a leading unknown."""
    if not spans:
        return []
    langs: list[str | None] = [lang for _, _, lang in spans]
    for i in range(1, len(langs)):
        if langs[i] is None:
            langs[i] = langs[i - 1]
    for i in range(len(langs) - 2, -1, -1):
        if langs[i] is None:
            langs[i] = langs[i + 1]

    merged: list[tuple[float, float, str | None]] = []
    for (start, end, _), lang in zip(spans, langs):
        if merged and merged[-1][2] == lang:
            merged[-1] = (merged[-1][0], end, lang)
        else:
            merged.append((start, end, lang))
    return merged
