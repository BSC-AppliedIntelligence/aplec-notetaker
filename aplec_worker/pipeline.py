from __future__ import annotations

import asyncio
import logging
import tempfile
import traceback
from pathlib import Path
from typing import Any

from .asr import WhisperAsrService
from .capture import MIC_AND_TAB
from .channels import load_channels, mix_for_listening, write_wav
from .config import Settings
from .diarization import NemotronDiarizer, PyannoteDiarizer
from .llm import LlamaCppClient
from .persistence import RunStore
from .voiceprints import identify_page_groups, identify_turns
from .meeting_chat import attach_chat
from .speaker_merge import (
    assignment_fallback_names,
    drop_nonspeaker_events,
    events_to_named_intervals,
    fuse_turns,
    group_turns_into_blocks,
    map_speakers_to_names,
    move_turns,
    page_groups,
    relabel_segments,
    same_person,
    words_to_speaker_segments,
)
from .schemas import (
    MeetingCompletedPayload,
    MeetingInsights,
    TranscriptResult,
    VexaMediaFile,
    VexaRecording,
)
from .vexa_client import RecordingLookupError, VexaClient

logger = logging.getLogger(__name__)

MIX_FILE = "mix.wav"


def _sanitize_transcript(transcript):
    """Final anti-hallucination pass, applied on EVERY path into the LLM and
    persistence (live runs, fallback single-pass, reprocess of old runs):
    collapse decoder loops, drop known stock fabrications, dedup near-identical
    text transcribed twice from overlapping diarization turns, and rebuild the
    flat text from the surviving segments."""
    from .textfilters import collapse_loops, dedup_overlapping_segments, is_fabrication

    if not transcript.segments:
        return transcript
    cleaned = []
    for seg in transcript.segments:
        text = collapse_loops((seg.text or "").strip())
        if not text:
            continue
        if is_fabrication(text, seg.language or transcript.language):
            logger.info("Sanitize: dropping fabricated segment %r", text[:60])
            continue
        if text != seg.text:
            seg = seg.model_copy(update={"text": text})
        cleaned.append(seg)
    before = list(cleaned)
    cleaned = dedup_overlapping_segments(cleaned)
    if len(cleaned) < len(before):
        # Key on IDENTITY, not (start, text): the duplicates this filter removes
        # are near-identical by definition, so a value key reported "dropped 0"
        # in exactly the case the log exists for.
        survivors = {id(s) for s in cleaned}
        gone = [s for s in before if id(s) not in survivors]
        # Name what was removed: a bare counter made it impossible to tell an
        # overlap echo from deleted speech, which is exactly the judgement this
        # filter can get wrong.
        logger.info(
            "Sanitize: dropped %d duplicate overlapping segment(s): %s",
            len(gone),
            "; ".join(
                f"{s.speaker or '?'} {s.start_seconds or 0.0:.1f}-{s.end_seconds or 0.0:.1f}s "
                f"{s.text[:40]!r}"
                for s in gone[:8]
            ),
        )
    if not cleaned:
        return transcript
    return transcript.model_copy(update={
        "segments": cleaned,
        "text": " ".join(s.text for s in cleaned),
    })


def load_waveform(audio_path: Path):
    """The recording as 16 kHz mono float32, decoded once per run and shared by
    the silence guard, the diarizer, voiceprints and the language split: each
    used to run its own ffmpeg decode of the whole file."""
    from .audio_io import load_waveform_16k

    return load_waveform_16k(audio_path)


def active_speech_seconds(waveform) -> float:
    """Seconds of speech-energy audio in the decoded recording."""
    from .audio_io import waveform_active_seconds

    return waveform_active_seconds(waveform)


class MeetingProcessor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.asr = WhisperAsrService(settings)
        self.llm = LlamaCppClient(settings)
        self.store = RunStore(settings)
        self.vexa = VexaClient(settings)
        self.diarizer = PyannoteDiarizer(settings)
        self.nemotron = NemotronDiarizer(settings)
        from .lid import EcapaLanguageId

        self.lid = EcapaLanguageId(settings)
        # Lazy: constructed on the first meeting where enrolled prints exist.
        self._voiceprint_backend = None
        # Whisper + pyannote share one GPU and one cached pipeline each; concurrent
        # hooks (e.g. replay double-fires) must queue here or memory spikes can OOM
        # the co-located llama server.
        self._gpu_lock = asyncio.Lock()

    def _meeting_recordings_from_hook(self, payload: MeetingCompletedPayload) -> list[VexaRecording]:
        raw_recordings = payload.meeting.data.get("recordings", [])
        return [VexaRecording.model_validate(recording) for recording in raw_recordings]

    @staticmethod
    def _pick_audio_media_file(recording: VexaRecording) -> VexaMediaFile | None:
        audio_media = [
            media for media in recording.media_files
            if (media.type or "").lower().startswith("audio")
        ]
        if audio_media:
            return audio_media[0]
        return recording.media_files[0] if recording.media_files else None

    @staticmethod
    def _sort_recordings(recordings: list[VexaRecording]) -> list[VexaRecording]:
        def key(recording: VexaRecording):
            return (
                (recording.status or "").lower() == "completed",
                recording.completed_at or "",
                recording.created_at or "",
                str(recording.id),
            )

        return sorted(recordings, key=key, reverse=True)

    def _resolve_local_recording_path(self, media_file: VexaMediaFile) -> Path | None:
        if not media_file.storage_path:
            return None

        recordings_root = self.settings.recordings_dir.resolve()

        def contained(candidate: Path) -> Path | None:
            # Hooks are network input; never resolve a recording outside recordings_dir.
            try:
                resolved = candidate.resolve(strict=True)
            except (OSError, RuntimeError):
                return None
            if not resolved.is_relative_to(recordings_root):
                return None
            return resolved

        storage_path = Path(media_file.storage_path)
        if storage_path.is_absolute():
            found = contained(storage_path)
            if found:
                return found
            # Container path (e.g. /data/recordings/...): re-root every candidate tail
            # under recordings_dir, with and without the matched "recordings" segment,
            # so both nested (recordings/recordings/<id>) and flat layouts resolve.
            parts = storage_path.parts
            for index, part in enumerate(parts):
                if part != "recordings":
                    continue
                for tail in (parts[index:], parts[index + 1 :]):
                    if not tail:
                        continue
                    found = contained(self.settings.recordings_dir / Path(*tail))
                    if found:
                        return found

        return contained(self.settings.recordings_dir / Path(str(storage_path).lstrip("/")))

    async def _settle_local_recording(self, path: Path, media_file: VexaMediaFile) -> Path:
        """Vexa's completion webhook can fire mid-finalization: storage_path may still
        point at the LAST CHUNK (e.g. 000003.webm) while the finalizer is assembling
        master.webm next to it. Wait (bounded) for the finalized master; fall back to
        whatever converges with the hook's file_size_bytes, else the largest sibling."""
        expected = media_file.file_size_bytes or 0
        if path.name.startswith("master.") or (expected and path.stat().st_size >= expected):
            return path

        deadline = asyncio.get_event_loop().time() + self.settings.recording_settle_seconds
        while True:
            masters = sorted(path.parent.glob("master.*"))
            if masters:
                master = masters[0]
                size = master.stat().st_size
                # settled = reached the announced size, or stable across one poll
                if not expected or size >= expected * 0.98:
                    logger.info("Recording settled to finalized %s (%d bytes)", master.name, size)
                    return master
                await asyncio.sleep(2)
                if master.stat().st_size == size:
                    logger.info("Recording master size stable at %d bytes; using it", size)
                    return master
                continue
            if asyncio.get_event_loop().time() >= deadline:
                siblings = [p for p in path.parent.iterdir() if p.is_file()]
                largest = max(siblings, key=lambda p: p.stat().st_size, default=path)
                logger.warning(
                    "Recording never settled to a master file; proceeding with largest sibling %s",
                    largest.name,
                )
                return largest
            await asyncio.sleep(3)

    async def _resolve_recording_input(
        self,
        payload: MeetingCompletedPayload,
        run_dir: Path,
    ) -> tuple[Path, VexaRecording, VexaMediaFile]:
        recordings = self._sort_recordings(self._meeting_recordings_from_hook(payload))

        if self.settings.prefer_local_recordings:
            for recording in recordings:
                media_file = self._pick_audio_media_file(recording)
                if not media_file:
                    continue
                local_path = self._resolve_local_recording_path(media_file)
                if local_path:
                    local_path = await self._settle_local_recording(local_path, media_file)
                    return local_path, recording, media_file

        api_recordings = await self.vexa.list_recordings(payload.meeting.id)
        for recording in self._sort_recordings(api_recordings):
            media_file = self._pick_audio_media_file(recording)
            if not media_file:
                continue

            local_path = self._resolve_local_recording_path(media_file)
            if local_path:
                return local_path, recording, media_file

            suffix = media_file.format or "bin"
            output_path = run_dir / "inputs" / f"recording.{suffix}"
            downloaded_path = await self.vexa.download_media_file(recording, media_file, output_path)
            return downloaded_path, recording, media_file

        raise RecordingLookupError(f"Could not resolve an audio recording for meeting {payload.meeting.id}")

    async def process(self, payload: MeetingCompletedPayload) -> dict[str, Any]:
        run_dir = self.store.prepare_run_dir(payload.meeting.id)
        self.store.write_json(run_dir / "hook.json", payload.model_dump(mode="json"))

        mix_dir = None
        try:
            audio_path, recording, media_file = await self._resolve_recording_input(payload, run_dir)
            mix_dir = await asyncio.to_thread(self._mix_channels, payload, audio_path)
            heard = Path(mix_dir.name) / MIX_FILE if mix_dir else audio_path
            waveform = None
            if self.settings.min_active_speech_seconds > 0:
                waveform = await asyncio.to_thread(load_waveform, heard)
                active_seconds = await asyncio.to_thread(active_speech_seconds, waveform)
                if active_seconds < self.settings.min_active_speech_seconds:
                    return self._persist_no_speech(
                        run_dir, payload, recording, media_file, audio_path, active_seconds
                    )
            multilingual = (
                self.settings.diarization_enabled
                and self.settings.asr_multilingual
                and self.settings.asr_backend == "whispercpp"
            )
            async with self._gpu_lock:
                if multilingual:
                    transcript = await self._transcribe_multilingual(
                        payload, media_file, heard, waveform
                    )
                    if transcript is None:  # degraded: fall back to single-pass
                        transcript, words = await asyncio.to_thread(self.asr.transcribe, heard, True)
                        if words:
                            transcript = await self._diarize(
                                payload, media_file, heard, transcript, words, waveform
                            )
                else:
                    want_words = self.settings.diarization_enabled
                    transcript, words = await asyncio.to_thread(self.asr.transcribe, heard, want_words)
                    if want_words and words:
                        transcript = await self._diarize(
                            payload, media_file, heard, transcript, words, waveform
                        )
                # ~230 MB per hour of audio: not needed by the summary.
                waveform = None
                await asyncio.to_thread(self._release_gpu_cache)
            transcript = attach_chat(transcript, payload.meeting.data)
            self.store.write_json(run_dir / "transcript.json", transcript.model_dump(mode="json"))
            # Before the summary: the transcript is already worth sharing, and
            # a failing summarizer must not keep it from the participants.
            self._auto_share(payload, transcript)

            return await self._summarize_and_persist(
                run_dir,
                payload,
                transcript,
                recording=recording,
                media_file=media_file,
                audio_path=audio_path,
            )
        except Exception as exc:
            self._write_error(run_dir, exc)
            raise
        finally:
            if mix_dir:
                mix_dir.cleanup()

    def _mix_channels(self, payload: MeetingCompletedPayload, audio_path: Path):
        """A recording with the microphone and the tab apart becomes the one signal
        the pipeline listens to, in a temporary directory the caller cleans up. None
        for every other recording, and when this fails: ffmpeg then downmixes the
        file, which loses nothing but the choice."""
        if payload.meeting.data.get("audio_channels") != MIC_AND_TAB:
            return None
        if not self.settings.audio_channels_enabled:
            return None
        directory = tempfile.TemporaryDirectory(prefix="aplec-mix-")
        try:
            mic, tab = load_channels(audio_path)
            mixed, match = mix_for_listening(mic, tab)
            write_wav(Path(directory.name) / MIX_FILE, mixed)
            if match.heard_twice:
                logger.info(
                    "Channels: the tab repeats the microphone %.2fs later on %d of %d windows; "
                    "listening to the tab", match.lag_seconds, match.matched, match.windows,
                )
            else:
                logger.info(
                    "Channels: different sound on each (%d of %d windows match); summing them",
                    match.matched, match.windows,
                )
            return directory
        except Exception:
            logger.exception("Could not use the channels apart; downmixing %s", audio_path.name)
            directory.cleanup()
            return None

    @staticmethod
    def _release_gpu_cache() -> None:
        """Return CUDA allocator cache to the OS. On this unified-memory host the
        worker otherwise retains tens of GB after word-level ASR + diarization,
        starving the co-located llama server."""
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    async def _speaker_turns(self, audio_path, min_speakers, decoded) -> tuple[list, list | None]:
        """pyannote's turns, and when Nemotron is on and usable, their combination
        with Nemotron's (fuse_turns), for the caller to choose from.

        pyannote takes the page's identities as a lower bound and Nemotron cannot,
        so it can fuse people: on a real meeting with 8 identities it heard 4 voices
        where pyannote kept 7. Fewer voices than identities means that happened, and
        there is no combination to offer; the same when Nemotron fails. It has 8
        speaker channels, so a meeting with 9 identities always ends here too."""
        turns = await asyncio.to_thread(self.diarizer.diarize, audio_path, min_speakers, decoded)
        if not self.settings.diarization_nemotron or not turns:
            return turns, None
        nemotron = await asyncio.to_thread(self.nemotron.diarize, decoded)
        if not nemotron:
            return turns, None
        voices = len({turn.speaker for turn in nemotron})
        if min_speakers and voices < min_speakers:
            logger.info(
                "Nemotron heard %d voice(s) for %d identities; keeping pyannote's turns",
                voices, min_speakers,
            )
            return turns, None
        fused = fuse_turns(nemotron, turns)
        logger.info(
            "Diarization: Nemotron's %d voices, %d turns; %d piece(s) from pyannote's %d voices",
            voices, len(nemotron), len(fused) - len(nemotron), len({t.speaker for t in turns}),
        )
        return turns, fused

    async def _name_speakers(self, turns, named, waveform, platform, owner_id) -> tuple[dict, list]:
        """Who each diarization label is: the page's vote, then voiceprints, then the
        Teams assignment. Returns label->name and every voice recognized."""
        mapping = map_speakers_to_names(turns, named)
        # Voice vote: enrolled voiceprints name the clusters the DOM vote
        # cannot see (hybrid-room voices) and re-unite oversplit clusters.
        voice_matches: list = []
        if waveform is not None:
            additions, voice_matches = await asyncio.to_thread(
                self._voice_identify, waveform, turns, mapping, named, owner_id
            )
            mapping.update(additions)
        mapping.update(assignment_fallback_names(turns, named, mapping, platform))
        return mapping, voice_matches

    async def _reassign_turns(self, turns, named, mapping, voice_matches, waveform):
        """Names go to whole clusters, so what diarization mixed into a cluster takes
        its name. Two passes give people their turns back: the turns where their tile
        was lit and the group's voice is theirs (speaker_merge.page_groups), then any
        turn whose voice alone is someone else in the meeting (voiceprints.
        identify_turns). On a real room this brought back a remote participant who
        had lost their name, and their task with it in the notes. Never raises; on
        failure the turns stay as they were."""
        if waveform is None or not (self.settings.page_voices_enabled or self.settings.voice_sweep_enabled):
            return turns, mapping, voice_matches
        try:
            tools = await asyncio.to_thread(self._voice_tools)
            if tools is None:
                return turns, mapping, voice_matches
            prints, backend = tools
            # A group is seconds of audio where a cluster is minutes, and it takes
            # speech from someone: automatic sharing's bar, the stricter of the two.
            gates = (
                max(self.settings.voiceprint_threshold, self.settings.auto_share_min_score),
                self.settings.voiceprint_margin,
            )
            reassigned = turns, mapping, voice_matches
            if self.settings.page_voices_enabled:
                groups = page_groups(reassigned[0], named, reassigned[1])
                accepted = await asyncio.to_thread(
                    identify_page_groups, waveform, groups, reassigned[1], prints, backend, *gates
                )
                reassigned = self._move(reassigned, accepted)
            if self.settings.voice_sweep_enabled:
                present = [*reassigned[1].values(), *{interval.name for interval in named}]
                accepted = await asyncio.to_thread(
                    identify_turns, waveform, reassigned[0], reassigned[1], present,
                    prints, backend, *gates,
                )
                reassigned = self._move(reassigned, accepted)
            return reassigned
        except Exception:
            logger.exception("Reassigning turns failed; keeping them as they were")
            return turns, mapping, voice_matches

    def _voice_tools(self):
        """The enrolled prints and the embedding backend, or None without them."""
        if not self.settings.voiceprints_enabled:
            return None
        from .voiceprints import VoiceprintStore

        store = VoiceprintStore(self.settings.data_dir)
        if self._voiceprint_backend is None and not store.load_all():
            return None
        backend = self.get_voiceprint_backend()
        prints = store.load_all(model=backend.model_id)
        return (prints, backend) if prints else None

    def _move(self, reassigned, accepted) -> tuple:
        turns, mapping, voice_matches = reassigned
        if not accepted:
            return reassigned
        moved, renamed = move_turns(
            turns, [(entry["group"], entry["name"]) for entry in accepted], mapping
        )
        return moved, renamed, self._with_new_matches(voice_matches, accepted, renamed)

    @staticmethod
    def _with_new_matches(voice_matches, accepted, mapping) -> list:
        """Someone only these turns recognized counts as recognized by voice, so
        automatic sharing reaches them."""
        from .schemas import VoiceMatch

        matches = list(voice_matches)
        seen = {match.name for match in matches}
        for entry in accepted:
            name = entry["name"]
            if name in seen:
                continue
            seen.add(name)
            same = [e for e in accepted if e["name"] == name]
            matches.append(VoiceMatch(
                speaker=next(label for label, owner in mapping.items() if same_person(owner, name)),
                name=name,
                email=entry["email"],
                score=entry["score"],
                seconds=round(sum(e["group"].seconds for e in same), 1),
            ))
        return matches

    async def _transcribe_multilingual(self, payload, media_file, audio_path, decoded=None):
        """Diarize first, then transcribe each same-speaker block with its own
        language detection — code-switched meetings come out as spoken instead of
        pinned to the language of the opening seconds. Returns None on failure
        (caller falls back to the single-pass path). `decoded`: the recording
        already decoded by the silence guard, if it ran."""
        from .schemas import TranscriptResult, TranscriptSegment

        try:
            if decoded is None:
                decoded = await asyncio.to_thread(load_waveform, audio_path)
            events = drop_nonspeaker_events(payload.meeting.data.get("speaker_events") or [])
            # Each distinct Meet identity that spoke is >= one acoustic voice,
            # so it is a safe LOWER bound for clustering (a hybrid room only
            # adds voices beyond its single identity). Guards against fusing
            # two people into one SPEAKER_N. The bound only holds for identities
            # that are PEOPLE: a DOM artifact counted as one forces pyannote to
            # invent a cluster, splitting a real voice in two.
            identities = {
                e.get("participant_name")
                for e in events
                if isinstance(e, dict) and e.get("participant_name")
            }
            min_speakers = len(identities) or None
            if min_speakers and min_speakers > 1:
                logger.info("Diarization hint: min_speakers=%d (distinct identities)", min_speakers)
            turns, fused = await self._speaker_turns(audio_path, min_speakers, decoded)

            duration = None
            if media_file is not None and media_file.duration_seconds:
                duration = float(media_file.duration_seconds)
            elif turns:
                duration = turns[-1].end
            named = events_to_named_intervals(events, duration)

            waveform = None
            if self.settings.asr_lid_split_enabled or self.settings.voiceprints_enabled:
                waveform = decoded

            platform, owner_id = payload.meeting.platform, payload.meeting.user_id
            mapping, voice_matches = await self._name_speakers(turns, named, waveform, platform, owner_id)
            if fused is not None:
                # The combination must not cost anyone their name. In a room with
                # more voices than Nemotron's 8 channels it folds people together:
                # on a real meeting pyannote named 8 people, and with Nemotron three
                # of them vanished into an unnamed 19-minute speaker. The count of
                # pyannote's own voices cannot tell this from the rooms the
                # combination fixes (10 voices for 8 people), the names can.
                fused_mapping, fused_matches = await self._name_speakers(
                    fused, named, waveform, platform, owner_id
                )
                lost = sorted(set(mapping.values()) - set(fused_mapping.values()))
                if lost:
                    logger.info("Keeping pyannote's turns: with Nemotron's, %s lose their name", lost)
                else:
                    turns, mapping, voice_matches = fused, fused_mapping, fused_matches
            turns, mapping, voice_matches = await self._reassign_turns(
                turns, named, mapping, voice_matches, waveform
            )

            blocks = group_turns_into_blocks(turns)
            if not blocks:
                logger.warning("Multilingual ASR: no speech blocks from diarization; falling back")
                return None

            # batch_session: every span call below shares one loaded whisper.cpp
            # model (whisper_batch.py) instead of one whisper-cli per call.
            with tempfile.TemporaryDirectory(prefix="aplec-ml-") as tmp, self.asr.batch_session():
                wav = await asyncio.to_thread(
                    self.asr.prepare_wav, audio_path, Path(tmp) / "full.wav"
                )

                # Phase A: all (start, end, speaker, pinned_lang) spans, temporal order.
                # Spans without real speech energy are dropped BEFORE decoding:
                # Whisper fabricates stock phrases on near-silent slivers, and a
                # skipped span is the only output it cannot corrupt.
                from .audio_io import SAMPLE_RATE, span_energy_threshold, waveform_active_seconds

                gate_s = self.settings.asr_min_span_speech_seconds
                gate_rms = None
                if waveform is not None and gate_s > 0:
                    gate_rms = span_energy_threshold(
                        waveform, self.settings.asr_span_gate_relative_db
                    )
                    logger.info("Span energy gate: RMS > %.5f", gate_rms)
                all_spans: list[list] = []
                for block in blocks:
                    for span_start, span_end, span_lang in await self._language_spans(
                        wav, waveform, block
                    ):
                        if waveform is not None and gate_s > 0:
                            sl = waveform[int(span_start * SAMPLE_RATE):int(span_end * SAMPLE_RATE)]
                            required = min(gate_s, 0.5 * (span_end - span_start))
                            # The measured defect here was the DISCARDED tail,
                            # not the window size: a 1.9s span whose last 0.9s
                            # was loud speech read 0.00s. waveform_active_seconds
                            # now measures the trailing partial window, which
                            # fixes it. A finer window was tried and rejected:
                            # the 1s RMS OVERSTATES sparse speech (0.4s of speech
                            # in 2s reads 2.0s), so a fine window would make this
                            # gate stricter and could drop short real
                            # interjections — an effect nobody has measured.
                            active = waveform_active_seconds(sl, rms_threshold=gate_rms)
                            if active < required:
                                logger.info(
                                    "Skipping span %.1f-%.1fs: below speech-energy gate "
                                    "(%.2fs active < %.2fs required)",
                                    span_start, span_end, active, required,
                                )
                                continue
                        all_spans.append([span_start, span_end, block.speaker, span_lang])
                all_spans.sort(key=lambda s: s[0])

                # Phase B: transcribe LONG spans — Whisper's LID is trustworthy
                # there, and they establish the meeting's language inventory.
                # Short spans are deferred: their 107-way auto-detect is near
                # chance (hallucinated sv/pl/pt in a pure EN/TR meeting).
                evidence_s = self.settings.asr_lang_evidence_seconds
                rescore = (
                    self.settings.lid_rescore_enabled and waveform is not None
                )
                results: dict[int, tuple[str, str | None]] = {}
                for idx, (span_start, span_end, _speaker, span_lang) in enumerate(all_spans):
                    if not rescore or (span_end - span_start) >= evidence_s:
                        results[idx] = await asyncio.to_thread(
                            self.asr.transcribe_block, wav, span_start, span_end, span_lang
                        )

                inventory: list[str] = []
                if rescore:
                    from .lid import LangSpan, language_inventory

                    inventory = language_inventory(
                        [
                            LangSpan(all_spans[i][0], all_spans[i][1], all_spans[i][2], results[i][1])
                            for i in results
                        ],
                        evidence_s,
                    )
                    if inventory:
                        logger.info("Language inventory (spans ≥%.0fs): %s", evidence_s, inventory)
                long_inventory = list(inventory)

                # Phase C1: Whisper's own read of the deferred short spans
                # (auto-detect + transcription), BEFORE any repair decisions.
                deferred = {idx for idx in range(len(all_spans)) if idx not in results}
                for idx in sorted(deferred):
                    span_start, span_end, _speaker, _lang = all_spans[idx]
                    results[idx] = await asyncio.to_thread(
                        self.asr.transcribe_block, wav, span_start, span_end, None
                    )

                # Widen the inventory with CUMULATIVE cross-span evidence: a
                # language whisper keeps reading on substantial spans is in the
                # meeting even if no single span reaches evidence_s. Field
                # failure: English spoken in 7.9s+6.9s spans of a Portuguese
                # meeting stayed off-inventory and the ECAPA repair (Brazilian-
                # accented English reads as 'pt') made Whisper re-transcribe =
                # TRANSLATE it. The cumulative rule subsumes the single-span
                # rule, so the inventory only ever grows here.
                added: list[str] = []
                if rescore:
                    widened = language_inventory(
                        [
                            LangSpan(all_spans[i][0], all_spans[i][1], all_spans[i][2], results[i][1])
                            for i in results
                        ],
                        evidence_s,
                    )
                    added = [lang for lang in widened if lang not in inventory]
                    if added:
                        logger.info(
                            "Inventory widened by cumulative evidence: +%s → %s", added, widened
                        )
                        inventory = widened
                main_lang = self._main_language(all_spans, results)
                unheard = await self._unheard_languages(waveform, all_spans, results, added, main_lang)
                unheard_long = await self._unheard_languages(
                    waveform, all_spans, results, long_inventory, main_lang, cumulative=False
                )
                speaker_langs: dict = {}
                unsure: set[int] = set()
                nobodys: set[str] = set()
                if unheard_long:
                    unsure = await self._unsure_spans(wav, all_spans, results, unheard_long)
                    speaker_langs = self._speaker_languages(all_spans, results, evidence_s, unsure)
                    nobodys = {
                        lang for lang in unheard_long
                        if not any(lang in spoken for spoken in speaker_langs.values())
                    }
                    logger.info(
                        "Speaker languages (%d span(s) Whisper is unsure of left out): %s%s",
                        len(unsure), {mapping.get(k, k): v for k, v in speaker_langs.items()},
                        f"; nobody spoke {sorted(nobodys)} at length" if nobodys else "",
                    )

                def first_tag(lang: str | None, speaker) -> tuple[str, bool] | None:
                    """The language to decode first, with `lang` as the second tag, when
                    `lang` is one the voice model does not hear, and whether it is the
                    speaker's own; None keeps the decode as it is. A language a long span
                    let in stays the language of whoever spoke it at length; one that
                    nobody did is a misread, and someone with nothing long gets the main
                    language first, as for a language the cumulative rule let in."""
                    if lang in unheard:
                        return main_lang, False
                    spoken = speaker_langs.get(speaker)
                    if lang in unheard_long and spoken and lang not in spoken:
                        return spoken[0], True
                    if lang in nobodys:
                        return main_lang, False
                    return None

                # Phase C2: walk spans in temporal order. Whisper is the
                # PRIMARY detector — it reads words, so when it lands on a
                # language that's actually in the meeting, trust it. ECAPA only
                # REPAIRS Whisper's out-of-inventory hallucinations (Turkish
                # heard as sv/pl), constrained to the inventory with a
                # same-speaker stickiness prior. Forcing a correct in-inventory
                # Whisper detection to another language makes Whisper *translate*
                # (English "Ah, I can hear you" → Turkish "Evet, duydum").
                prev_any: str | None = None
                prev_by_speaker: dict = {}
                kept: list[list] = []
                # The label each kept span had before an unheard language gave way to
                # the main one, for the second language pass's shares.
                share_labels: list[str | None] = []
                main_first: set[int] = set()
                by_speaker = by_nobody = 0
                for idx, (span_start, span_end, speaker, span_lang) in enumerate(all_spans):
                    text, lang = results[idx]
                    heard_as = None
                    own_first = False
                    if idx in deferred and inventory and lang not in inventory:
                        try:
                            from .lid import confirm_out_of_inventory, pick_language

                            probs = await asyncio.to_thread(
                                self.lid.probs, waveform, span_start, span_end
                            )
                            aside = confirm_out_of_inventory(
                                lang, inventory, probs,
                                span_end - span_start,
                                self.settings.lid_escape_min_prob,
                                self.settings.lid_escape_min_seconds,
                            )
                            if not aside:
                                choice = pick_language(
                                    probs, inventory, prev_by_speaker.get(speaker), prev_any
                                )
                                logger.info(
                                    "Repair %.1f-%.1fs: whisper=%s (off-inventory) → %s",
                                    span_start, span_end, lang, choice,
                                )
                                if first_tag(choice, speaker):
                                    lang = choice  # decoded below, with two tags
                                elif choice and choice != lang:
                                    text, lang = await asyncio.to_thread(
                                        self.asr.transcribe_block, wav, span_start, span_end, choice
                                    )
                            else:
                                logger.info(
                                    "Aside %.1f-%.1fs: whisper=%s confirmed by ecapa (%.2f)",
                                    span_start, span_end, lang, probs.get(lang, 0.0),
                                )
                        except Exception:
                            logger.exception(
                                "LID rescoring failed for span %.1f-%.1fs; keeping Whisper auto",
                                span_start,
                                span_end,
                            )
                    # A long span keeps the language it was read in, unless Whisper's own
                    # language ID is unsure of it (_unsure_spans) and its speaker never
                    # spoke that language at length: then it is decoded like a short one.
                    tag = first_tag(lang, speaker) if idx in deferred or idx in unsure else None
                    if tag and text.strip():
                        # Whisper read a language the voice model does not hear, so
                        # this is most likely speech in the main language decoded in
                        # the wrong one, i.e. translated. Main language first (the
                        # speaker's own, for a language a long span let in), the read
                        # one second: the second tag keeps a real aside (an English
                        # title read aloud) in its own words. A span that came back
                        # empty stays empty: decoded again, noise became "The city is
                        # also home to the city of San Francisco" on Miami.
                        first, own_first = tag
                        decoded, _ = await asyncio.to_thread(
                            self.asr.transcribe_block, wav, span_start, span_end, first, lang
                        )
                        if idx in deferred or self._usable(decoded, first, lang):
                            heard_as, text, lang = lang, decoded, first
                        else:
                            # Seconds of real speech: what was read beats nothing.
                            own_first = False
                            logger.info(
                                "Long span %.1f-%.1fs: the decode with %s first came back empty "
                                "or fabricated; it stays %s",
                                span_start, span_end, first, lang,
                            )
                    text = text.strip()
                    if not text:
                        # Whisper returned nothing for a span that passed the
                        # energy gate — including a temperature retry that came
                        # back empty (looks_loopy("") is False, so it is always
                        # accepted). Silent until now.
                        logger.info(
                            "Empty transcription for span %.1f-%.1fs (%s); span dropped",
                            span_start, span_end, lang or "auto",
                        )
                        continue
                    from .textfilters import collapse_loops, is_fabrication

                    text = collapse_loops(text)
                    if is_fabrication(text, lang) or (heard_as and is_fabrication(text, heard_as)):
                        logger.info(
                            "Dropping fabricated span %.1f-%.1fs (%s): %s",
                            span_start, span_end, lang, text[:60],
                        )
                        continue
                    if lang:
                        # The language as read, so the repair keeps choosing as it was
                        # measured; what it chooses in an unheard one is decoded as above.
                        prev_any = heard_as or lang
                        prev_by_speaker[speaker] = heard_as or lang
                    if heard_as:
                        main_first.add(len(kept))
                        by_speaker += own_first
                        by_nobody += heard_as in nobodys and not own_first
                    share_labels.append(heard_as or lang)
                    kept.append([span_start, span_end, speaker, lang, text])
                if unheard:
                    logger.info(
                        "Unheard %s: %d span(s) decoded with %s first and labeled %s",
                        sorted(unheard), len(main_first) - by_speaker - by_nobody, main_lang, main_lang,
                    )
                if unheard_long:
                    logger.info(
                        "Unheard %s, by speaker: %d span(s) of speakers who never spoke it at "
                        "length decoded with their own language first, %d of speakers with "
                        "nothing long with %s first",
                        sorted(unheard_long), by_speaker, by_nobody, main_lang,
                    )

                await self._second_language_pass(wav, kept, share_labels, main_first)
                segments = [
                    TranscriptSegment(
                        start_seconds=span_start,
                        end_seconds=span_end,
                        text=text,
                        speaker=mapping.get(speaker, speaker),
                        language=lang,
                    )
                    for span_start, span_end, speaker, lang, text in kept
                ]

            if not segments:
                logger.warning("Multilingual ASR produced no segments; falling back")
                return None

            lang_seconds: dict[str, float] = {}
            for seg in segments:
                if seg.language:
                    lang_seconds[seg.language] = lang_seconds.get(seg.language, 0.0) + (
                        (seg.end_seconds or 0) - (seg.start_seconds or 0)
                    )
            dominant = max(lang_seconds, key=lang_seconds.get) if lang_seconds else None
            languages = sorted(lang_seconds, key=lang_seconds.get, reverse=True)
            logger.info(
                "Multilingual ASR: %d blocks, languages=%s, mapping=%s",
                len(segments), languages, mapping or "{}",
            )
            return TranscriptResult(
                text=" ".join(seg.text for seg in segments),
                language=dominant,
                segments=segments,
                speaker_names=mapping,
                voice_matches=voice_matches,
            )
        except Exception:
            logger.exception("Multilingual ASR failed; falling back to single-pass")
            return None

    @staticmethod
    def _usable(text: str, *languages: str | None) -> bool:
        """What the span phases would keep of a decode: something left after the loop
        collapse that is no stock fabrication in any of `languages`."""
        from .textfilters import collapse_loops, is_fabrication

        text = collapse_loops(text.strip())
        return bool(text) and not any(lang and is_fabrication(text, lang) for lang in languages)

    # Shorter spans keep their single language; the tests never went below this.
    _SECOND_LANGUAGE_MIN_SECONDS = 2.5

    @staticmethod
    def _main_language(all_spans: list[list], results: dict) -> str | None:
        """The language Whisper detected on most of the speech, before any repair."""
        seconds: dict[str, float] = {}
        for idx, (start, end, _speaker, _lang) in enumerate(all_spans):
            lang = results[idx][1] if idx in results else None
            if lang:
                seconds[lang] = seconds.get(lang, 0.0) + end - start
        return max(seconds, key=seconds.get) if seconds else None

    # A language is heard when the voice model gives it this much on half of
    # the seconds of its evidence (see _unheard_languages).
    _HEARD_MIN_PROBABILITY = 0.5
    _HEARD_MIN_SHARE = 0.5

    async def _unheard_languages(
        self,
        waveform,
        all_spans: list[list],
        results: dict,
        added: list[str],
        main: str | None,
        cumulative: bool = True,
    ) -> set[str]:
        """Languages of `added` that the voice model does not hear: by default the ones
        the cumulative rule alone let in, with cumulative=False the ones long spans did.

        Whisper reads an accent as another language: in one meeting a speaker's
        accented Spanish came out as Portuguese on 52 s of 4-8 s spans, enough
        for the cumulative rule, and from then on every
        short span Whisper read as Portuguese was trusted and TRANSLATED into it; two
        English titles read aloud let English in the same way, and short Spanish
        turns came out as "Thank you." ECAPA on the same
        evidence: none of those languages reached 0.5 on any second of it, while real
        Catalan spoken in short turns reached it on 79-100% of its seconds. The
        language stays in the inventory, so the repair still routes speech to it, but
        its spans are decoded with the main language first and take its label.
        End to end on that meeting, against the remote copy of the same call: words
        found 12.4% → 26.7% on the segments it changes, 64.0% → 67.6% over the 45 min
        both hold; Miami cpWER 42.6 → 42.3 on the four recordings with a cumulative
        language. Without the voice check it translated real Catalan into Spanish,
        and for every short span outside the main language it translated real
        Spanish into Catalan on Parlament (29.8 → 30.4).

        A long span can let a wrong language in as well: in another meeting one 13.6 s
        span of Spanish read as Welsh, and the repair then sent 9 short spans there,
        translated into Welsh. But ECAPA hears accents, not languages, and on long
        spans it calls real languages unheard: English read aloud by a participant (0%),
        the Spanish or English of bilingual speakers in 9 of Miami's 37 recordings. So an
        unheard long-span language only changes the spans of speakers who never spoke it
        at length (first_tag in _transcribe_multilingual): their short spans, and the
        long ones Whisper itself is unsure of (_unsure_spans); whoever did keeps it, and
        the second tag is the language as read. One that nobody spoke at length is a
        misread, and a speaker with nothing long gets the main language first. On Miami's
        references that
        rule touches 28 spans (21 misreads, 3 real switches) where every short span
        outside the main language would be 3030 (2089 real switches). End to end:
        Miami cpWER 37.44 → 37.44 on the 5 recordings it touches, AMI headsets 20.98 →
        20.93, room 28.73 → 28.60, Corts WER 49.2 → 49.0; read on two real meetings,
        77 spans, nearly all came back in their speaker's language, a few lost a word
        or stayed noise."""
        enabled = (
            self.settings.asr_confirm_cumulative_languages
            if cumulative
            else self.settings.asr_speaker_languages
        )
        if not added or not main or not enabled:
            return set()
        if not self.asr.second_language_available():
            # Pinning the main language alone would translate a real aside.
            return set()
        unheard: set[str] = set()
        for lang in added:
            if lang == main:
                continue
            evidence = [
                (start, end) for idx, (start, end, _speaker, _l) in enumerate(all_spans)
                if idx in results and results[idx][1] == lang and end - start >= 4.0
            ]
            total = sum(end - start for start, end in evidence)
            if not total:
                continue
            heard = 0.0
            try:
                for start, end in evidence:
                    probs = await asyncio.to_thread(self.lid.probs, waveform, start, end)
                    if probs.get(lang, 0.0) >= self._HEARD_MIN_PROBABILITY:
                        heard += end - start
            except Exception:
                logger.exception("Voice check of %s failed; it stays a language of the meeting", lang)
                continue
            share = heard / total
            if share < self._HEARD_MIN_SHARE:
                unheard.add(lang)
            if lang not in unheard:
                verdict = "kept"
            elif cumulative:
                verdict = f"unheard, its spans get {main} first"
            else:
                verdict = "unheard, it stays with whoever spoke it at length"
            logger.info(
                "%s %s: the voice model hears it on %.0f%% of %.0fs of evidence; %s",
                "Cumulative" if cumulative else "Long-span", lang, share * 100, total, verdict,
            )
        return unheard

    # Whisper is sure of a span's language from this probability of its own language ID.
    _SURE_MIN_PROBABILITY = 0.5

    async def _unsure_spans(
        self, wav: Path, all_spans: list[list], results: dict, languages: set[str]
    ) -> set[int]:
        """Spans of 4 s or more read in one of `languages` whose own language ID
        (whisper -dl) does not give that language at _SURE_MIN_PROBABILITY: they do not
        make it a language of their speaker (_speaker_languages).

        A misread long span is what puts a fake language in someone's history, and
        Whisper is unsure of it there: the 13.6 s of Spanish read as Welsh scores
        cy 0.46, and the spans that gave one participant of another meeting Italian
        and Portuguese 0.40 and mostly under 0.5. Languages
        really spoken at length score 0.5 or more on 84-100% of their seconds: English
        read aloud 0.94-0.96, Catalan 0.83-0.99 in five real meetings, the real Spanish
        and English of every bilingual Miami recording, Parlament's Spanish. Two fakes
        pass (a fabricated "Terima kasih." at 0.51, a Corts Portuguese at 0.66) and one
        real span fails (Miami herring1, English and Spanish mixed in 9 s, 0.49). A
        language ID that fails keeps the span, as before.

        Such a span of 8 s or more is also decoded like a short one when its speaker
        never spoke the language at length: otherwise it kept the label, and the Welsh
        stayed on that meeting's page with its words already Spanish. On the public sets
        that changes 9 long spans with word errors about even (Corts 11007 → 10999,
        Miami cpWER -4, AMI room +5 on one span, whose Welsh-first decode had been the
        more literal); on real meetings two, the Welsh one (same words) and a
        9.7 s span of a Catalan speaker read as Italian, which comes out in the
        speaker's most spoken language, Spanish, as wrong as the Italian was."""
        unsure: set[int] = set()
        for idx, (start, end, _speaker, _lang) in enumerate(all_spans):
            lang = results[idx][1] if idx in results else None
            if lang not in languages or end - start < 4.0:
                continue
            got, prob = await asyncio.to_thread(self.asr.detect_language, wav, start, end)
            if got is not None and (got != lang or prob < self._SURE_MIN_PROBABILITY):
                unsure.add(idx)
        return unsure

    @staticmethod
    def _speaker_languages(
        all_spans: list[list], results: dict, evidence_s: float, unsure: set[int] | frozenset = frozenset()
    ) -> dict:
        """The languages each speaker spoke at length, most spoken first: the inventory's
        own rule (evidence_s over spans of 4 s or more, as Whisper read them) applied to
        one speaker's spans at a time, leaving out the `unsure` ones."""
        from .lid import LangSpan, language_inventory

        spans: dict = {}
        for idx, (start, end, speaker, _lang) in enumerate(all_spans):
            if idx in results and idx not in unsure:
                spans.setdefault(speaker, []).append(LangSpan(start, end, speaker, results[idx][1]))
        return {
            speaker: langs
            for speaker, own in spans.items()
            if (langs := language_inventory(own, evidence_s))
        }

    async def _second_language_pass(
        self,
        wav: Path,
        spans: list[list],
        share_labels: list[str | None] | None = None,
        skip: set[int] | frozenset = frozenset(),
    ) -> None:
        """Decode again, with a second language tag, the spans that can use one.

        `spans` holds final [start, end, speaker, language, text] entries and is
        updated in place. A pinned language makes Whisper TRANSLATE speech in any
        other language into it, and two tags (<|ca|><|es|>) largely stop that. The
        span's own language goes first, and its label stays what the language
        pipeline decided.

        Two cases, chosen from the meeting's own languages (no fixed list; a
        one-language meeting decodes nothing here):
        - a span outside the main language gets the main one: an aside stays in
          its language, and a misdetection (accented English read as Welsh) stops
          being translated into the language Whisper thought it heard;
        - a span IN the main language gets the meeting's second language, but only
          when that one holds asr_second_language_min_share of the speech. Without
          the condition the second tag rewords ordinary speech for nothing: on real
          Spanish meetings with a 2-9% Catalan share it moved 8-19% of the words and
          turned some real speech into "Gracias."; with real code-switching it pays.

        `skip` holds spans already decoded with two tags (an unheard language,
        see _unheard_languages), and `share_labels` the labels the shares are
        counted on: the ones those spans had before they took the main language,
        which is what this pass was measured with.
        """
        if not self.settings.asr_second_language or not self.asr.second_language_available():
            return
        labels = share_labels if share_labels is not None else [span[3] for span in spans]
        seconds: dict[str, float] = {}
        for (start, end, _speaker, _lang, _text), lang in zip(spans, labels):
            if lang:
                seconds[lang] = seconds.get(lang, 0.0) + end - start
        ranked = sorted(seconds, key=seconds.get, reverse=True)
        if len(ranked) < 2:
            return
        main, second = ranked[0], ranked[1]
        share = seconds[second] / sum(seconds.values())
        everywhere = share >= self.settings.asr_second_language_min_share
        decoded = changed = 0
        for n, span in enumerate(spans):
            start, end, _speaker, lang, text = span
            if n in skip or not lang or end - start < self._SECOND_LANGUAGE_MIN_SECONDS:
                continue
            if lang != main:
                other = main
            elif everywhere:
                other = second
            else:
                continue
            new_text, _ = await asyncio.to_thread(
                self.asr.transcribe_block, wav, start, end, lang, other
            )
            decoded += 1
            from .textfilters import collapse_loops, is_fabrication

            new_text = collapse_loops(new_text.strip())
            # The single-language decode already passed these filters: keep it
            # rather than lose the span.
            if not new_text or is_fabrication(new_text, lang):
                logger.info(
                    "Second language decode of span %.1f-%.1fs discarded: %s",
                    start, end, new_text[:60] or "empty",
                )
                continue
            changed += new_text != text
            span[4] = new_text
        logger.info(
            "Second language: main=%s, second=%s (%.1f%% of the speech), %s; "
            "%d span(s) decoded, %d changed",
            main, second, share * 100,
            "on every span" if everywhere else "only off the main language",
            decoded, changed,
        )

    _LID_MIN_PROBABILITY = 0.4
    # Above this, a block is worth splitting even if no pause clears the normal
    # threshold: one span this long means one coarse timestamp, dialogue
    # rendered out of order, and a single decode of several minutes.
    _LONG_BLOCK_SECONDS = 60.0

    async def _language_spans(self, wav: Path, waveform, block) -> list[tuple[float, float, str | None]]:
        """(start, end, pinned_language) spans for one speaker block.

        A speaker may switch languages mid-block (code-switching); one auto-detect
        per block would translate the post-switch speech into the opening language.
        Split the block at pauses, language-ID each chunk (fast encoder pass),
        re-merge adjacent same-language chunks, and pin the language per span.
        Low-confidence chunks inherit a neighbor's language. Falls back to one
        auto-detected span (language None) whenever splitting isn't worthwhile.
        """
        from .speaker_merge import find_pause_splits, merge_language_spans

        whole = [(block.start, block.end, None)]
        if (
            waveform is None
            # The knob was documented but never read here: the waveform is
            # loaded whenever voiceprints are on, so APLEC_ASR_LID_SPLIT=false
            # disabled neither the split nor the per-chunk LID passes.
            or not self.settings.asr_lid_split_enabled
            or block.end - block.start < self.settings.asr_lid_min_block_seconds
        ):
            return whole
        splits = find_pause_splits(
            waveform,
            block.start,
            block.end,
            min_pause=self.settings.asr_lid_min_pause_seconds,
            min_chunk=self.settings.asr_lid_min_chunk_seconds,
        )
        if not splits and (block.end - block.start) > self._LONG_BLOCK_SECONDS:
            # group_turns_into_blocks' max_block_seconds only stops turns from
            # being MERGED past the ceiling — a single pyannote turn of 188s
            # becomes one 188s block, and with no pause above the normal
            # threshold it becomes ONE span and ONE segment (field:
            # a 187.9s segment of 427 words). Retry with a
            # shorter pause so the cut still lands in silence, never mid-word.
            splits = find_pause_splits(
                waveform,
                block.start,
                block.end,
                min_pause=max(0.2, self.settings.asr_lid_min_pause_seconds / 2),
                min_chunk=max(self.settings.asr_lid_min_chunk_seconds, 10.0),
            )
            if splits:
                logger.info(
                    "Long block %.1f-%.1fs split at %d shorter pause(s)",
                    block.start, block.end, len(splits),
                )
            else:
                logger.info(
                    "Long block %.1f-%.1fs has no usable pause; staying one span",
                    block.start, block.end,
                )
        if not splits:
            return whole

        bounds = [block.start, *splits, block.end]
        chunks = list(zip(bounds, bounds[1:]))
        langs: list[str | None] = []
        probs: list[float] = []
        for chunk_start, chunk_end in chunks:
            lang, prob = await asyncio.to_thread(
                self.asr.detect_language, wav, chunk_start, chunk_end
            )
            langs.append(lang if prob >= self._LID_MIN_PROBABILITY else None)
            probs.append(prob)

        spans = merge_language_spans(
            [(s, e, lang) for (s, e), lang in zip(chunks, langs)]
        )
        if len(spans) <= 1 and (block.end - block.start) > self._LONG_BLOCK_SECONDS:
            # One language across the whole block, but the block is too long to
            # decode (and timestamp) as a single span: keep the pause cuts and
            # pin the single language to each. Without this the merge undoes the
            # split and the block returns as ONE span — the long-block fix was a
            # no-op until a self-audit caught it.
            single = spans[0][2] if spans else None
            logger.info(
                "Long block %.1f-%.1fs is single-language (%s); keeping %d pause cut(s)",
                block.start, block.end, single or "auto", len(chunks) - 1,
            )
            return [(s, e, single) for s, e in chunks]
        if len(spans) <= 1:
            # Single language after all: one pass, language pinned (skips re-detect).
            pinned = spans[0][2] if spans else None
            if pinned and (block.end - block.start) >= self.settings.asr_lang_evidence_seconds:
                # A pin on a span this long is NEVER revisited: the inventory and
                # the ECAPA repair only touch deferred SHORT spans. A wrong pin
                # makes Whisper translate, which corrupts content rather than
                # just the label — so log what was pinned and how confident the
                # per-chunk detections were, to make the risk measurable.
                logger.info(
                    "Pinned %s for whole block %.1f-%.1fs (%s), chunk p=%.2f-%.2f; not revisited",
                    pinned, block.start, block.end, block.speaker,
                    min(probs) if probs else 0.0,
                    max(probs) if probs else 0.0,
                )
            return [(block.start, block.end, pinned)]
        logger.info(
            "Language switch inside block %.1f-%.1fs (%s): %s",
            block.start,
            block.end,
            block.speaker,
            ", ".join(f"{lang or '?'} {s:.1f}-{e:.1f}s" for s, e, lang in spans),
        )
        return spans

    def get_voiceprint_backend(self):
        """Lazy, shared embedding backend (also used by the enroll endpoint)."""
        from .voiceprints import load_backend

        if self._voiceprint_backend is None:
            self._voiceprint_backend = load_backend(
                self.settings.voiceprint_backend,
                self.settings.voiceprint_model_dir or None,
            )
        return self._voiceprint_backend

    def _owner_emails(self, owner_id) -> list[str]:
        """The meeting owner's roster emails (normally one); empty if unreadable."""
        try:
            from .users import merged_users

            roster = merged_users(self.settings.web_users, self.settings.data_dir)
            return sorted(e for e, u in roster.items() if str(u.vexa_user_id) == str(owner_id))
        except Exception:
            logger.exception("Could not read the roster for the owner of a meeting")
            return []

    def _voice_identify(self, waveform, turns, mapping, named, owner_id) -> tuple[dict[str, str], list]:
        """Run the voiceprint vote; never raises (identification is optional).
        Returns the label->name assignments to apply and every recognized
        voice (VoiceMatch), the input to automatic sharing."""
        if not self.settings.voiceprints_enabled:
            return {}, []
        try:
            from .schemas import VoiceMatch
            from .voiceprints import VoiceprintStore, identify_clusters

            store = VoiceprintStore(self.settings.data_dir)
            if self._voiceprint_backend is None and not store.load_all():
                return {}, []
            backend = self.get_voiceprint_backend()
            prints = store.load_all(model=backend.model_id)
            if not prints:
                return {}, []
            # Who the meeting itself vouches for: a tile on the page, or its owner.
            page = {interval.name for interval in named}
            owners = set(self._owner_emails(owner_id))

            def on_page(print_) -> bool:
                email = str((print_.metadata or {}).get("email") or "").strip().lower()
                return email in owners or any(same_person(print_.name, name) for name in page)

            accepted: list[dict] = []
            additions = identify_clusters(
                waveform, turns, mapping, prints, backend,
                self.settings.voiceprint_threshold, self.settings.voiceprint_margin,
                override_dom=self.settings.voiceprint_override_dom,
                accepted=accepted,
                on_page=on_page,
                off_page_threshold=self.settings.voiceprint_off_page_threshold,
            )
            speech: dict[str, float] = {}
            for t in turns:
                speech[t.speaker] = speech.get(t.speaker, 0.0) + max(0.0, t.end - t.start)
            matches = [
                VoiceMatch(**m, seconds=round(speech.get(m["speaker"], 0.0), 1)) for m in accepted
            ]
            return additions, matches
        except Exception:
            logger.exception("Voiceprint identification failed; continuing without it")
            return {}, []

    def _auto_share(self, payload: MeetingCompletedPayload, transcript: TranscriptResult) -> None:
        """Share the meeting, read only, with the registered users recognized
        BY VOICE in it (on by default; each owner can
        turn it off in Settings). Voice only: the roster keeps no names, and a
        Meet display name is whatever someone typed. Stricter than naming a
        speaker (auto_share_min_score, auto_share_min_seconds), so a short or
        borderline match never opens a meeting to someone who was not there,
        and stricter again for someone with no tile on the page, whose voice is
        the only evidence (auto_share_off_page_min_score).
        Someone the owner removed is never added back. Never raises."""
        try:
            matches = transcript.voice_matches
            if not matches or not self.settings.auto_share_enabled:
                return
            from .meeting_access import auto_grant
            from .user_prefs import get_auto_share
            from .users import merged_users

            owner_id = payload.meeting.user_id
            meeting_id = str(payload.meeting.id)
            if not get_auto_share(self.settings.data_dir, owner_id):
                logger.info("Auto-share %s: turned off by its owner", meeting_id)
                return
            roster = merged_users(self.settings.web_users, self.settings.data_dir)
            owners = sorted(e for e, u in roster.items() if str(u.vexa_user_id) == str(owner_id))
            page = {
                e.get("participant_name")
                for e in payload.meeting.data.get("speaker_events") or []
                if isinstance(e, dict) and e.get("participant_name")
            }
            wanted: set[str] = set()
            for m in matches:
                email = (m.email or "").strip().lower()
                if not email or email in owners or email not in roster:
                    continue
                on_page = any(same_person(m.name, name) for name in page)
                min_score = self.settings.auto_share_min_score
                if not on_page:
                    min_score = max(min_score, self.settings.auto_share_off_page_min_score)
                if m.score < min_score or m.seconds < self.settings.auto_share_min_seconds:
                    logger.info(
                        "Auto-share %s: not sharing with %s (%s %.3f, %.0fs of speech%s; needs >=%.2f and >=%.0fs)",
                        meeting_id, email, m.speaker, m.score, m.seconds, "" if on_page else ", no tile",
                        min_score, self.settings.auto_share_min_seconds,
                    )
                    continue
                wanted.add(email)
            if not wanted:
                return
            added = auto_grant(
                self.settings.data_dir, meeting_id, sorted(wanted),
                granted_by=owners[0] if owners else "aplec",
            )
            if added:
                logger.info("Auto-share %s: shared with %s (recognized by voice)", meeting_id, ", ".join(added))
        except Exception:
            logger.exception("Automatic sharing failed for meeting %s", payload.meeting.id)

    async def _diarize(self, payload, media_file, audio_path, transcript, words, decoded=None):
        """Attach speaker labels (and real names when the hook carries speaker_events).
        Any failure degrades gracefully to the plain transcript."""
        try:
            events = drop_nonspeaker_events(payload.meeting.data.get("speaker_events") or [])
            identities = {
                e.get("participant_name")
                for e in events
                if isinstance(e, dict) and e.get("participant_name")
            }
            if decoded is None:
                decoded = await asyncio.to_thread(load_waveform, audio_path)
            turns = await asyncio.to_thread(
                self.diarizer.diarize, audio_path, len(identities) or None, decoded
            )
            duration = None
            if media_file is not None and media_file.duration_seconds:
                duration = float(media_file.duration_seconds)
            elif words:
                duration = words[-1].end
            named = events_to_named_intervals(events, duration)
            mapping = map_speakers_to_names(turns, named)
            voice_matches: list = []
            if self.settings.voiceprints_enabled:
                additions, voice_matches = await asyncio.to_thread(
                    self._voice_identify, decoded, turns, mapping, named, payload.meeting.user_id
                )
                mapping.update(additions)
            mapping.update(
                assignment_fallback_names(turns, named, mapping, payload.meeting.platform)
            )
            turns, mapping, voice_matches = await self._reassign_turns(
                turns, named, mapping, voice_matches, decoded
            )
            segments = relabel_segments(words_to_speaker_segments(words, turns), mapping)
            logger.info(
                "Diarization: %d turns, %d named events, mapping=%s",
                len(turns),
                len(events),
                mapping or "{}",
            )
            return transcript.model_copy(
                update={"segments": segments, "speaker_names": mapping, "voice_matches": voice_matches}
            )
        except Exception:
            logger.exception("Diarization failed; continuing with plain transcript")
            return transcript

    async def reprocess_with_transcript(
        self,
        payload: MeetingCompletedPayload,
        transcript: TranscriptResult,
    ) -> dict[str, Any]:
        run_dir = self.store.prepare_run_dir(payload.meeting.id)
        self.store.write_json(run_dir / "hook.json", payload.model_dump(mode="json"))
        # Before writing: Ask reads transcript.json, so the chat must be in it.
        transcript = attach_chat(transcript, payload.meeting.data)
        self.store.write_json(run_dir / "transcript.json", transcript.model_dump(mode="json"))
        self._auto_share(payload, transcript)

        try:
            return await self._summarize_and_persist(
                run_dir,
                payload,
                transcript,
                recording=None,
                media_file=None,
                audio_path=None,
            )
        except Exception as exc:
            self._write_error(run_dir, exc)
            raise

    def _write_error(self, run_dir: Path, exc: Exception) -> None:
        self.store.write_json(
            run_dir / "error.json",
            {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )

    def _persist_no_speech(
        self,
        run_dir: Path,
        payload: MeetingCompletedPayload,
        recording: VexaRecording | None,
        media_file: VexaMediaFile | None,
        audio_path: Path | None,
        active_seconds: float,
    ) -> dict[str, Any]:
        """Persist a run for a recording with no detectable speech, skipping
        ASR and LLM entirely (Whisper hallucinates on silence — the classic
        bot-joined-but-nobody-came meeting)."""
        # chat still counts: people may have typed even if nobody spoke
        transcript = attach_chat(TranscriptResult(text=""), payload.meeting.data)
        insights = MeetingInsights(
            summary="No speech detected in this meeting's recording."
        )
        self.store.write_json(run_dir / "transcript.json", transcript.model_dump(mode="json"))
        result = self._build_result(
            payload, recording, media_file, audio_path, transcript.model_dump(), insights, "no_speech"
        )
        self.store.write_json(run_dir / "result.json", result)
        self.store.update_latest(payload.meeting.id, result)
        logger.info(
            "Meeting %s recording has %.1fs of speech-energy audio (< %.1fs); "
            "persisted as quality=no_speech without ASR/LLM",
            payload.meeting.id,
            active_seconds,
            self.settings.min_active_speech_seconds,
        )
        return result

    async def _summarize_and_persist(
        self,
        run_dir: Path,
        payload: MeetingCompletedPayload,
        transcript: TranscriptResult,
        recording: VexaRecording | None,
        media_file: VexaMediaFile | None,
        audio_path: Path | None,
    ) -> dict[str, Any]:
        from .user_prefs import get_summary_language

        # Idempotent: a reprocess of an older run gets the payload's chat too.
        transcript = attach_chat(transcript, payload.meeting.data)
        transcript = _sanitize_transcript(transcript)
        output_language = get_summary_language(self.settings.data_dir, payload.meeting.user_id)
        llm_result = await self.llm.summarize(
            payload.meeting, transcript, output_language=output_language
        )

        (run_dir / "llm_raw.txt").write_text(llm_result.raw_response, encoding="utf-8")
        if llm_result.raw_repair_response is not None:
            (run_dir / "llm_raw_repair.txt").write_text(
                llm_result.raw_repair_response, encoding="utf-8"
            )
        if llm_result.raw_partial_responses:
            (run_dir / "llm_raw_partials.txt").write_text(
                "\n\n----- PARTIAL -----\n\n".join(llm_result.raw_partial_responses),
                encoding="utf-8",
            )

        insights = llm_result.insights
        quality = self._detect_quality(transcript, insights)
        if quality == "suspect":
            logger.warning(
                "Meeting %s LLM output flagged as suspect (transcript=%d chars, every insight list empty)",
                payload.meeting.id,
                len(transcript.text),
            )

        result = self._build_result(
            payload, recording, media_file, audio_path, transcript.model_dump(), insights, quality
        )
        result["llm_strategy"] = llm_result.strategy
        self.store.write_json(run_dir / "result.json", result)
        self.store.update_latest(payload.meeting.id, result)

        logger.info(
            "Processed meeting %s into %s (quality=%s)",
            payload.meeting.id,
            run_dir,
            quality,
        )
        return result

    @staticmethod
    def _detect_quality(transcript: TranscriptResult, insights: MeetingInsights) -> str:
        any_content = any([
            insights.most_important_points,
            insights.next_steps,
            insights.decisions,
            insights.action_items,
            insights.risks_or_open_questions,
        ])
        if len(transcript.text) > 500 and not any_content:
            return "suspect"
        return "ok"

    def _build_result(
        self,
        payload: MeetingCompletedPayload,
        recording: VexaRecording | None,
        media_file: VexaMediaFile | None,
        audio_path: Path | None,
        transcript: dict[str, Any],
        insights: MeetingInsights,
        quality: str,
    ) -> dict[str, Any]:
        return {
            "event": payload.event,
            "meeting": payload.meeting.model_dump(mode="json"),
            "recording": recording.model_dump(mode="json") if recording is not None else None,
            "media_file": media_file.model_dump(mode="json") if media_file is not None else None,
            "resolved_audio_path": str(audio_path) if audio_path is not None else None,
            "transcript": transcript,
            "insights": insights.model_dump(mode="json"),
            "quality": quality,
        }
