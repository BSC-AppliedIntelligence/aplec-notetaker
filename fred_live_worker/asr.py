from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
from contextlib import contextmanager
from pathlib import Path

from .config import Settings
from .schemas import TranscriptResult, TranscriptSegment
from .speaker_merge import SHORT_TURN_SECONDS, Word
from .whisper_batch import WhisperBatchProcess, WhisperBatchRequestError

logger = logging.getLogger(__name__)

# What a span under SHORT_TURN_SECONDS gets; see transcribe_block.
_SHORT_SPAN_PAD_SECONDS = 0.2
_SHORT_SPAN_MIN_LOGPROB = -0.6
_SHORT_SPAN_MAX_UNCHECKED_WORDS = 2


def _parse_whispercpp_json(data: dict, want_words: bool) -> tuple[TranscriptResult, list[Word]]:
    """Map whisper-cli -oj output to our transcript contract. Offsets are in ms.
    With want_words (-ml 1), every entry is one word -> words list + empty segments
    (speaker merge rebuilds segments); otherwise entries are sentence segments."""
    entries = data.get("transcription") or []
    language = (data.get("result") or {}).get("language")

    words: list[Word] = []
    segments: list[TranscriptSegment] = []
    parts: list[str] = []

    for entry in entries:
        text = (entry.get("text") or "").strip()
        offsets = entry.get("offsets") or {}
        start = offsets.get("from")
        end = offsets.get("to")
        if not text:
            continue
        parts.append(text)
        if want_words:
            if start is not None:
                words.append(
                    Word(start=start / 1000.0, end=(end if end is not None else start) / 1000.0, text=text)
                )
        else:
            segments.append(
                TranscriptSegment(
                    start_seconds=start / 1000.0 if start is not None else None,
                    end_seconds=end / 1000.0 if end is not None else None,
                    text=text,
                )
            )

    transcript = TranscriptResult(text=" ".join(parts), language=language, segments=segments)
    return transcript, words


def _avg_logprob(data: dict) -> float | None:
    """Mean log probability of the text tokens of whisper-cli -ojf output (special
    tokens read "[_BEG_]", "[_TT_12]"...): the number whisper-batch reports."""
    probabilities = [
        token["p"]
        for entry in data.get("transcription") or []
        for token in entry.get("tokens") or []
        if not token.get("text", "").startswith("[_")
    ]
    if not probabilities:
        return None
    return sum(math.log(max(p, 1e-6)) for p in probabilities) / len(probabilities)


class WhisperAsrService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipeline = None
        self._lock = threading.Lock()
        # whisper-batch process shared by the span calls of open batch_sessions.
        self._batch: WhisperBatchProcess | None = None
        self._batch_users = 0
        self._batch_lock = threading.Lock()

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            has_safetensors = any(self.settings.asr_model_dir.glob("*.safetensors"))

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                str(self.settings.asr_model_dir),
                low_cpu_mem_usage=True,
                torch_dtype=torch_dtype,
                use_safetensors=has_safetensors,
            )

            if torch.cuda.is_available():
                model = model.to(device)

            processor = AutoProcessor.from_pretrained(str(self.settings.asr_model_dir))
            self._pipeline = pipeline(
                task="automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                chunk_length_s=self.settings.asr_chunk_length_s,
                batch_size=self.settings.asr_batch_size,
                torch_dtype=torch_dtype,
                device=device,
            )

        return self._pipeline

    def transcribe(self, audio_path: Path, want_words: bool = False) -> tuple[TranscriptResult, list[Word]]:
        """Transcribe audio. With want_words=True, requests word-level timestamps
        (for diarization alignment) and returns them alongside a segment-less
        transcript — segments are rebuilt from words + speaker turns downstream."""
        if self.settings.asr_backend == "whispercpp":
            return self._transcribe_whispercpp(audio_path, want_words)
        return self._transcribe_transformers(audio_path, want_words)

    def _transcribe_whispercpp(
        self, audio_path: Path, want_words: bool
    ) -> tuple[TranscriptResult, list[Word]]:
        """GGML backend: ~4GB resident and seconds per meeting, vs the transformers
        word-timestamp path whose ~65GB spikes repeatedly OOM'd this host."""
        import subprocess
        import tempfile

        from .audio_io import SAMPLE_RATE

        with tempfile.TemporaryDirectory(prefix="fredlive-asr-") as tmp:
            wav_path = Path(tmp) / "audio.wav"
            ffmpeg = self._ffmpeg_bin()
            subprocess.run(
                [ffmpeg, "-y", "-i", str(audio_path), "-ar", str(SAMPLE_RATE), "-ac", "1",
                 "-c:a", "pcm_s16le", str(wav_path)],
                check=True, capture_output=True,
            )

            out_base = Path(tmp) / "out"
            cmd = [
                str(self.settings.whispercpp_bin),
                "-m", str(self.settings.whispercpp_model),
                "-f", str(wav_path),
                "-l", self.settings.asr_language or "auto",
                "-oj", "-of", str(out_base),
                "-np", "-sns",
            ]
            if want_words:
                cmd += ["-ml", "1", "-sow"]
            subprocess.run(cmd, check=True, capture_output=True)

            data = json.loads((out_base.with_suffix(".json")).read_text(encoding="utf-8"))

        return _parse_whispercpp_json(data, want_words)

    def prepare_wav(self, audio_path: Path, out_path: Path) -> Path:
        """Decode any input to 16k mono pcm wav once (blocks are sliced from it)."""
        import subprocess

        from .audio_io import SAMPLE_RATE

        subprocess.run(
            [self._ffmpeg_bin(), "-y", "-i", str(audio_path), "-ar", str(SAMPLE_RATE),
             "-ac", "1", "-c:a", "pcm_s16le", str(out_path)],
            check=True, capture_output=True,
        )
        return out_path

    def _batch_binary(self) -> Path | None:
        if self.settings.asr_backend != "whispercpp" or not self.settings.asr_whispercpp_batch:
            return None
        binary = self.settings.whispercpp_batch_bin or self.settings.whispercpp_bin.with_name(
            "whisper-batch"
        )
        return binary if binary.is_file() and os.access(binary, os.X_OK) else None

    @contextmanager
    def batch_session(self):
        """Keep one whisper.cpp model loaded for the transcribe_block and
        detect_language calls made inside this block (see whisper_batch.py).
        Without the whisper-batch binary they run whisper-cli, as before."""
        with self._batch_lock:
            self._batch_users += 1
            if self._batch is None and self._batch_users == 1:
                binary = self._batch_binary()
                if binary is not None:
                    batch = WhisperBatchProcess(binary, self.settings.whispercpp_model)
                    try:
                        batch.spawn()
                        self._batch = batch
                    except OSError as exc:
                        logger.warning("whisper-batch did not start (%s); using whisper-cli", exc)
        try:
            yield
        finally:
            with self._batch_lock:
                self._batch_users -= 1
                batch = None
                if self._batch_users == 0:
                    batch, self._batch = self._batch, None
            if batch is not None:
                if batch.calls:
                    logger.info(
                        "whisper-batch: %d calls in %.1fs (model loaded once, in %.1fs)",
                        batch.calls, batch.busy_seconds, batch.load_seconds or 0.0,
                    )
                batch.close()

    def _batch_call(self, payload: dict) -> dict | None:
        """whisper-batch's answer, or None when whisper-cli must run the call:
        no session open, the process rejected this request, or the process
        failed (then whisper-cli runs every later call of the session too)."""
        batch = self._batch
        if batch is None:
            return None
        try:
            return batch.request(payload)
        except WhisperBatchRequestError as exc:
            logger.warning("whisper-batch rejected a %s call (%s); whisper-cli runs it",
                           payload.get("kind"), exc)
            return None
        except Exception as exc:
            logger.warning(
                "whisper-batch stopped answering (%s; stderr: %s); whisper-cli runs the "
                "remaining calls", exc, batch.stderr_tail() or "-",
            )
            with self._batch_lock:
                if self._batch is batch:
                    self._batch = None
            batch.close()
            return None

    # The two engines for one slice, kept side by side: the worker uses
    # whisper-batch and falls back to whisper-cli, and
    # scripts/verify_whisper_batch.py compares them through these very methods.

    def _whisper_cli_decode(
        self, slice_path: Path, language: str | None, temperature: float | None
    ) -> dict:
        """whisper-cli's -ojf JSON for one slice, plus the avg_logprob
        whisper-batch reports (-ojf carries the tokens it is computed from)."""
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory(prefix="fredlive-cli-") as tmp:
            out_base = Path(tmp) / "out"
            extra = ["-tp", str(temperature)] if temperature is not None else []
            subprocess.run(
                [str(self.settings.whispercpp_bin), "-m", str(self.settings.whispercpp_model),
                 "-f", str(slice_path), "-l", language or "auto",
                 "-ojf", "-of", str(out_base), "-np", "-sns", *extra],
                check=True, capture_output=True,
            )
            data = json.loads(out_base.with_suffix(".json").read_text(encoding="utf-8"))
            data["avg_logprob"] = _avg_logprob(data)
            return data

    def _whisper_cli_detect(self, slice_path: Path) -> str:
        """whisper-cli -dl output; the detected language is only in its log."""
        import subprocess

        proc = subprocess.run(
            [str(self.settings.whispercpp_bin), "-m", str(self.settings.whispercpp_model),
             "-f", str(slice_path), "-dl"],
            capture_output=True, text=True,
        )
        return (proc.stdout or "") + (proc.stderr or "")

    @staticmethod
    def _batch_decode_request(
        slice_path: Path,
        language: str | None,
        temperature: float | None,
        second_language: str | None = None,
    ) -> dict:
        """The whisper-batch request for what _whisper_cli_decode runs (-sns always,
        -tp only on the loop retry). The second language is whisper-batch's own."""
        request = {
            "kind": "transcribe", "path": str(slice_path),
            "language": language or "auto", "suppress_nst": True,
        }
        if temperature is not None:
            request["temperature"] = temperature
        if second_language:
            request["second_language"] = second_language
        return request

    @staticmethod
    def _batch_json(answer: dict) -> dict:
        """A whisper-batch answer in whisper-cli's JSON shape, so both are
        parsed by the same code (_parse_whispercpp_json)."""
        return {
            "result": {"language": answer.get("language")},
            "transcription": [{"text": t} for t in answer.get("texts") or []],
            "avg_logprob": answer.get("avg_logprob"),
        }

    def second_language_available(self) -> bool:
        """Whether transcribe_block can use second_language right now: whisper-cli
        has no such option, and a whisper-batch built without the patch neither."""
        batch = self._batch
        return batch is not None and batch.second_language

    def transcribe_block(
        self,
        wav_path: Path,
        start: float,
        end: float,
        language: str | None = None,
        second_language: str | None = None,
    ) -> tuple[str, str | None]:
        """Transcribe one time-slice (whispercpp backend only). With language=None
        the block gets its own auto-detection; passing a language pins the decoder
        (used after intra-block language-switch detection — auto would re-detect on
        the opening seconds only). Returns (text, language).

        second_language puts a second tag after the pinned language in the decoder
        prompt, so speech in that language is not translated into the first (see
        MeetingProcessor._second_language_pass). whisper-cli cannot do this: if it
        has to run the call, the language stays pinned alone, as it always was.

        Repetition loops ("no, no, no..." x14) are a boundary-sensitive decoder
        instability: the same audio transcribes cleanly or loops depending on
        span edges (verified A/B on a real meeting). When the greedy decode
        loops, retry once with temperature sampling; if both loop, keep the
        retry (collapse_loops downstream trims it).

        A span under SHORT_TURN_SECONDS gets a little audio on each side, since
        pyannote's edges clip words this short. On a fraction of a second Whisper
        also writes the likeliest sentence rather than what it heard, so several
        words need a confident decoder or the span comes back empty (below -0.6:
        13% of the words real on AMI, 28% on NOTSOFAR). One or two words pass
        regardless: real answers score that low too ("Vale", "Sí", "Sim", "No"),
        and a wrong backchannel costs little."""
        import tempfile

        from .textfilters import looks_loopy

        short = end - start < SHORT_TURN_SECONDS
        pad = _SHORT_SPAN_PAD_SECONDS if short else 0.0
        with tempfile.TemporaryDirectory(prefix="fredlive-blk-") as tmp:
            slice_path = self._slice_wav(
                wav_path, max(0.0, start - pad), end + pad, Path(tmp) / "slice.wav"
            )

            def run(temperature: float | None) -> tuple[str, str | None, float | None]:
                answer = self._batch_call(
                    self._batch_decode_request(slice_path, language, temperature, second_language)
                )
                if answer is not None:
                    data = self._batch_json(answer)
                else:
                    data = self._whisper_cli_decode(slice_path, language, temperature)
                transcript, _ = _parse_whispercpp_json(data, want_words=False)
                return transcript.text, transcript.language or language, data.get("avg_logprob")

            text, lang, logprob = run(None)
            if self.settings.asr_loop_retry and looks_loopy(text):
                logger.info(
                    "Loop detected in span %.1f-%.1fs; retrying with temperature", start, end
                )
                retry_text, retry_lang, retry_logprob = run(0.4)
                # Compare DISTINCT words: the looped decode is long but
                # repetitive, so a total-word comparison would reject the clean
                # retry we actually want. Distinct content is what must not
                # shrink — looks_loopy needs 8 tokens, so without this guard a
                # retry of "Sim." replaced a 40s span (and an empty one always
                # won, since looks_loopy("") is False). The looped decode is
                # worth more than that: collapse_loops trims the repetition
                # downstream and the real words around it survive.
                distinct_before = len(set(text.split()))
                distinct_after = len(set(retry_text.split()))
                if not retry_text.strip():
                    logger.info(
                        "Loop retry returned nothing for span %.1f-%.1fs; keeping the looped decode",
                        start, end,
                    )
                elif distinct_after * 2 < distinct_before:
                    logger.info(
                        "Loop retry lost content on span %.1f-%.1fs (%d -> %d distinct words); "
                        "keeping the looped decode",
                        start, end, distinct_before, distinct_after,
                    )
                elif not looks_loopy(retry_text) or distinct_after > distinct_before:
                    text, lang, logprob = retry_text, retry_lang, retry_logprob
        if (
            short
            and logprob is not None
            and logprob < _SHORT_SPAN_MIN_LOGPROB
            and len(text.split()) > _SHORT_SPAN_MAX_UNCHECKED_WORDS
        ):
            logger.info(
                "Dropping short span %.1f-%.1fs: decoder confidence %.2f (%s): %s",
                start, end, logprob, lang, text.strip()[:40],
            )
            return "", lang
        return text, lang

    _LID_RE = re.compile(r"auto-detected language: (\w+) \(p = ([0-9.]+)\)")

    def detect_language(self, wav_path: Path, start: float, end: float) -> tuple[str | None, float]:
        """Fast language ID of a time-slice via whisper.cpp -dl (encoder pass only,
        <1s). NOTE: -np would suppress the detection line — don't add it. Returns
        (language, probability); (None, 0.0) when detection fails."""
        import tempfile

        with tempfile.TemporaryDirectory(prefix="fredlive-lid-") as tmp:
            slice_path = self._slice_wav(wav_path, start, end, Path(tmp) / "slice.wav")
            answer = self._batch_call({"kind": "detect", "path": str(slice_path)})
            if answer is not None:
                # whisper-batch returns the very log line whisper-cli prints.
                output = answer.get("log") or ""
            else:
                output = self._whisper_cli_detect(slice_path)
        match = self._LID_RE.search(output)
        if not match:
            return None, 0.0
        return match.group(1), float(match.group(2))

    def _slice_wav(self, wav_path: Path, start: float, end: float, out_path: Path) -> Path:
        import subprocess

        subprocess.run(
            [self._ffmpeg_bin(), "-y", "-ss", f"{max(0.0, start):.3f}",
             "-t", f"{max(0.1, end - start):.3f}", "-i", str(wav_path),
             "-c:a", "pcm_s16le", str(out_path)],
            check=True, capture_output=True,
        )
        return out_path

    @staticmethod
    def _ffmpeg_bin() -> str:
        import shutil

        found = shutil.which("ffmpeg")
        if found:
            return found
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()

    def _transcribe_transformers(
        self, audio_path: Path, want_words: bool
    ) -> tuple[TranscriptResult, list[Word]]:
        pipe = self._get_pipeline()

        generate_kwargs = {"task": "transcribe"}
        if self.settings.asr_language and self.settings.asr_language.lower() != "auto":
            generate_kwargs["language"] = self.settings.asr_language

        from .audio_io import SAMPLE_RATE, load_waveform_16k

        result = pipe(
            {"raw": load_waveform_16k(audio_path), "sampling_rate": SAMPLE_RATE},
            return_timestamps="word" if want_words else True,
            generate_kwargs=generate_kwargs,
        )

        segments: list[TranscriptSegment] = []
        words: list[Word] = []
        for chunk in result.get("chunks", []):
            timestamp = chunk.get("timestamp") or (None, None)
            text = (chunk.get("text") or "").strip()
            if want_words:
                if text and timestamp[0] is not None:
                    end = timestamp[1] if timestamp[1] is not None else timestamp[0]
                    words.append(Word(start=float(timestamp[0]), end=float(end), text=text))
            else:
                segments.append(
                    TranscriptSegment(
                        start_seconds=timestamp[0],
                        end_seconds=timestamp[1],
                        text=text,
                    )
                )

        transcript = TranscriptResult(
            text=(result.get("text") or "").strip(),
            language=self.settings.asr_language,
            segments=segments,
        )
        return transcript, words

