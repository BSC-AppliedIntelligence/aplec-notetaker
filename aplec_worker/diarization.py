from __future__ import annotations

import json
import logging
import math
import subprocess
import threading
from pathlib import Path

from .config import REPO_ROOT, Settings
from .speaker_merge import SpeakerTurn

logger = logging.getLogger(__name__)


class PyannoteDiarizer:
    """Lazy-loaded pyannote/speaker-diarization-3.1 wrapper.

    The pipeline is constructed on first use and cached (same pattern as
    WhisperAsrService). Audio is decoded via ffmpeg to a 16k mono waveform so
    webm/opus inputs work regardless of torchaudio backend support.
    """

    # pyannote.audio 4.x resolves the 3.1 pipeline through this current-gen model;
    # it is gated separately from speaker-diarization-3.1 on HF.
    MODEL_ID = "pyannote/speaker-diarization-community-1"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipeline = None
        self._lock = threading.Lock()

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            import torch
            from pyannote.audio import Pipeline

            token: str | bool = True  # True -> use cached HF login
            if self.settings.hf_token:
                token = self.settings.hf_token.get_secret_value()

            pipeline = Pipeline.from_pretrained(self.MODEL_ID, token=token)
            if pipeline is None:
                raise RuntimeError(
                    f"Could not load {self.MODEL_ID} — has the HF gate been accepted "
                    "for the token in use (APLEC_HF_TOKEN or cached login)?"
                )
            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
            if self.settings.diarization_skip_silent:
                install_skip_silent(pipeline)
            self._pipeline = pipeline

        return self._pipeline

    @staticmethod
    def _load_waveform(audio_path: Path, waveform=None):
        import torch

        from .audio_io import SAMPLE_RATE, load_waveform_16k

        if waveform is None:
            waveform = load_waveform_16k(audio_path)
        else:
            # pyannote gets its own copy: the caller keeps reading this array
            # (voiceprints, language split, energy gate) after diarization.
            waveform = waveform.copy()
        return {"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": SAMPLE_RATE}

    def diarize(
        self, audio_path: Path, min_speakers: int | None = None, waveform=None
    ) -> list[SpeakerTurn]:
        """min_speakers: lower bound on acoustic speakers. Each distinct Meet
        identity that produced speaker events is at least one voice (a hybrid
        room can only ADD voices beyond its one identity), so passing that
        count guards against under-clustering (two people fused into one
        SPEAKER_N). Never pass max: acoustic speakers can exceed identities.

        waveform: the recording already decoded by load_waveform_16k, when the
        caller has it, so the file is not decoded once more."""
        pipeline = self._get_pipeline()
        kwargs = {}
        if min_speakers and min_speakers > 1:
            kwargs["min_speakers"] = int(min_speakers)
        result = pipeline(self._load_waveform(audio_path, waveform), **kwargs)
        # pyannote 4.x wraps the Annotation in a DiarizeOutput; 3.x returned it bare.
        annotation = getattr(result, "speaker_diarization", result)
        turns = [
            SpeakerTurn(start=float(segment.start), end=float(segment.end), speaker=str(label))
            for segment, _, label in annotation.itertracks(yield_label=True)
        ]
        return sorted(turns, key=lambda t: t.start)


class NemotronDiarizer:
    """Nemotron 3 Diarization through tools/nemotron/diarize.py, in its own Python
    environment (scripts/setup_nemotron.sh): the model needs transformers 5 and the
    worker holds transformers below 5. A process per meeting: 4 s to load, 4-9 s per
    meeting of up to an hour.

    It can only help. Whatever goes wrong returns None and pyannote's turns stand."""

    TOOL = REPO_ROOT / "tools" / "nemotron" / "diarize.py"
    TIMEOUT_SECONDS = 900.0

    def __init__(self, settings: Settings):
        self.settings = settings

    def diarize(self, waveform) -> list[SpeakerTurn] | None:
        python, model = self.settings.nemotron_python, self.settings.nemotron_model_dir
        if not python.is_file() or not model.is_dir():
            logger.warning("Nemotron: %s or %s is missing (scripts/setup_nemotron.sh)", python, model)
            return None
        try:
            done = subprocess.run(
                [str(python), str(self.TOOL), str(model), "--threshold", str(self.settings.nemotron_threshold)],
                input=waveform.astype("float32").tobytes(),
                capture_output=True,
                timeout=self.TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("Nemotron did not run: %s", str(exc)[:200])
            return None
        if done.returncode != 0:
            logger.warning(
                "Nemotron failed (code %d): %s",
                done.returncode, done.stderr.decode("utf-8", "replace")[-300:],
            )
            return None
        try:
            answer = json.loads(done.stdout.decode("utf-8", "replace").strip().splitlines()[-1])
            turns = [SpeakerTurn(float(a), float(b), str(c)) for a, b, c in answer["turns"]]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            logger.warning("Nemotron answered something unreadable: %s", exc)
            return None
        logger.info(
            "Nemotron: %d voices, %d turns in %.1fs",
            len({turn.speaker for turn in turns}), len(turns), answer.get("seconds") or 0.0,
        )
        return sorted(turns, key=lambda t: t.start)


# Skipping the embeddings of silent speakers.
#
# pyannote embeds every (10 s chunk, local speaker) pair, three per chunk,
# including speakers who are silent for the whole chunk. Their mask is all
# zeros, the weighted statistics pooling then yields exactly zero, and the
# embedding is the last layer's bias: one constant vector, whatever the audio.
# Measured on real meetings: 42-46% of the pairs are silent, and embeddings are
# 97% of the diarization time (the GB10 is slow at FP32 convolutions), so the
# embedding step went 33.3 -> 19.2 s and 61.7 -> 33.4 s.
#
# Not bit-identical: the pairs that are computed meet other batch neighbours,
# which moves their values in the last float bits (~1e-6). pyannote's own
# embeddings already shift that much when a batch's composition changes; the
# turns built on them were identical on every meeting compared.
#
# Rejected for the same step: TF32 (1.4x) and bf16 (2.5x). pyannote disables
# TF32 on purpose because it measured lower accuracy (pyannote-audio #1370).


def get_embeddings_skip_silent(self, file, binary_segmentations, exclude_overlap=False, hook=None):
    """pyannote.audio 4.0's SpeakerDiarization.get_embeddings (MIT licence,
    (c) CNRS), except that pairs whose mask is all zeros are not run through
    the model: they get the constant a zero mask produces."""
    import numpy as np
    import torch
    from pyannote.core import SlidingWindowFeature

    duration = binary_segmentations.sliding_window.duration
    num_chunks, num_frames, num_speakers = binary_segmentations.data.shape

    if exclude_overlap:
        # minimum number of samples needed to extract an embedding
        min_num_samples = self._embedding.min_num_samples
        num_samples = duration * self._embedding.sample_rate
        min_num_frames = math.ceil(num_frames * min_num_samples / num_samples)
        # zero-out frames with overlapping speech
        clean_frames = 1.0 * (np.sum(binary_segmentations.data, axis=2, keepdims=True) < 2)
        clean_segmentations = SlidingWindowFeature(
            binary_segmentations.data * clean_frames, binary_segmentations.sliding_window
        )
    else:
        min_num_frames = -1
        clean_segmentations = SlidingWindowFeature(
            binary_segmentations.data, binary_segmentations.sliding_window
        )

    jobs = []  # (index into the flat (chunk, speaker) grid, chunk, mask used)
    for c, ((chunk, masks), (_, clean_masks)) in enumerate(
        zip(binary_segmentations, clean_segmentations)
    ):
        # mask may contain NaN (in case of partial stitching)
        masks = np.nan_to_num(masks, nan=0.0).astype(np.float32)
        clean_masks = np.nan_to_num(clean_masks, nan=0.0).astype(np.float32)
        for s, (mask, clean_mask) in enumerate(zip(masks.T, clean_masks.T)):
            used_mask = clean_mask if np.sum(clean_mask) > min_num_frames else mask
            if np.any(used_mask):
                jobs.append((c * num_speakers + s, chunk, used_mask))

    silent = _silent_embedding(self, duration, num_frames)
    embeddings = np.repeat(silent[None], num_chunks * num_speakers, axis=0)

    batch_size = self.embedding_batch_size
    batch_count = math.ceil(len(jobs) / batch_size)
    if hook is not None:
        hook("embeddings", None, total=batch_count, completed=0)
    for i in range(batch_count):
        batch = jobs[i * batch_size:(i + 1) * batch_size]
        waveforms = [self._audio.crop(file, chunk, mode="pad")[0][None] for _, chunk, _ in batch]
        masks = [torch.from_numpy(used_mask)[None] for _, _, used_mask in batch]
        embedding_batch = self._embedding(torch.vstack(waveforms), masks=torch.vstack(masks))
        embeddings[[index for index, _, _ in batch]] = embedding_batch
        if hook is not None:
            hook("embeddings", embedding_batch, total=batch_count, completed=i + 1)

    return embeddings.reshape(num_chunks, num_speakers, silent.shape[-1])


def _silent_embedding(pipeline, duration: float, num_frames: int, waveform=None):
    """What the embedding model returns for a chunk whose mask is all zeros."""
    import torch

    num_samples = int(round(duration * pipeline._embedding.sample_rate))
    if waveform is None:
        waveform = torch.zeros(1, 1, num_samples)
    return pipeline._embedding(waveform, masks=torch.zeros(1, num_frames))[0]


def install_skip_silent(pipeline) -> bool:
    """Swap in get_embeddings_skip_silent when it is safe: the pipeline is the
    pyannote 4.0 SpeakerDiarization it was copied from, and its embedding model
    really returns one constant for a silent chunk, checked on two different
    waveforms (a future model whose silent output depends on the audio, or that
    returns NaN for it, keeps the stock method)."""
    import types

    import numpy as np
    import pyannote.audio
    import torch
    from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization

    version = getattr(pyannote.audio, "__version__", "?")
    if type(pipeline) is not SpeakerDiarization or not version.startswith("4.0."):
        logger.info("Diarization: stock embeddings (pyannote %s, %s)", version, type(pipeline).__name__)
        return False
    try:
        model = pipeline._segmentation.model
        duration = float(model.specifications.duration)
        num_frames = model.num_frames(int(round(duration * model.hparams.sample_rate)))
        num_samples = int(round(duration * pipeline._embedding.sample_rate))
        noise = torch.Generator().manual_seed(0)
        silent = _silent_embedding(pipeline, duration, num_frames)
        a = _silent_embedding(pipeline, duration, num_frames,
                              torch.randn(1, 1, num_samples, generator=noise) * 0.1)
        b = _silent_embedding(pipeline, duration, num_frames,
                              torch.randn(1, 1, num_samples, generator=noise) * 0.1)
    except Exception:
        logger.exception("Diarization: could not check the silent-chunk embedding; stock embeddings")
        return False
    if np.isnan(silent).any() or not (np.array_equal(silent, a) and np.array_equal(silent, b)):
        logger.info("Diarization: silent-chunk embedding is not one constant; stock embeddings")
        return False
    pipeline.get_embeddings = types.MethodType(get_embeddings_skip_silent, pipeline)
    logger.info("Diarization: embeddings of silent speakers skipped (constant vector)")
    return True
