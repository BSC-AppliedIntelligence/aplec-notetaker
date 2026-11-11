"""Single audio decode path for the worker.

Decodes any container/codec (webm/opus included) to 16 kHz mono float32 via the
ffmpeg *binary* (system ffmpeg or the imageio-ffmpeg fallback on PATH), bypassing
torchcodec/torchaudio backends entirely — torchcodec requires FFmpeg shared
libraries that this host does not have, and transformers silently prefers it for
path inputs when it is importable (it arrives as a pyannote dependency).
Both Whisper and pyannote consume the array produced here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
# The span gate's fixed threshold (~-40 dBFS), which is also waveform_active_seconds' default.
SPAN_GATE_RMS = 0.01
# Lowest threshold a recording can bring the span gate to, and what counts as audible when
# measuring its level: digital silence must stay below both.
SPAN_GATE_MIN_RMS = 2e-4
_AUDIBLE_RMS = 1e-4


def load_waveform_16k(audio_path: Path) -> np.ndarray:
    from transformers.pipelines.audio_utils import ffmpeg_read

    return ffmpeg_read(audio_path.read_bytes(), SAMPLE_RATE)


def span_energy_threshold(
    waveform: np.ndarray,
    relative_db: float = 30.0,
    sample_rate: int = SAMPLE_RATE,
) -> float:
    """RMS threshold of the per-span gate, scaled to this recording's level.

    SPAN_GATE_RMS is an absolute level, and a recording's level depends on the
    microphone. The extension's captures sit far above it (median 0.17-0.34, not
    one block under it in a month of real meetings), but AMI's room recordings at
    their original level lost whole turns: 182 spans skipped in 3 meetings, cpWER
    54.8 against 25.1 with this threshold, which matches turning the gate off. So
    the threshold follows the recording, `relative_db` below its loud speech (90th
    percentile of the audible 1s windows, so a door slam does not move it), and
    never rises above SPAN_GATE_RMS: a loud recording keeps the fixed gate.
    relative_db <= 0 keeps the fixed gate everywhere.
    """
    if relative_db <= 0 or waveform.size < sample_rate:
        return SPAN_GATE_RMS
    whole = waveform.size // sample_rate
    frames = waveform[: whole * sample_rate].reshape(whole, sample_rate)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    audible = rms[rms > _AUDIBLE_RMS]
    level = float(np.percentile(audible, 90)) if audible.size else 0.0
    return max(SPAN_GATE_MIN_RMS, min(SPAN_GATE_RMS, level * 10 ** (-relative_db / 20)))


def waveform_active_seconds(
    waveform: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    rms_threshold: float = SPAN_GATE_RMS,
    window_seconds: float = 1.0,
) -> float:
    """Seconds of windows whose RMS exceeds the threshold.

    A cheap speech-energy proxy: a bot alone in a meeting records digital
    silence (RMS ~0), while any real speech pushes windows well above 0.01
    (~-40 dBFS). Used to skip ASR/LLM entirely on no-speech recordings, where
    Whisper otherwise hallucinates filler like repeated "Thank you."

    `window_seconds` is the measurement resolution, and it matters on SHORT
    audio: with the 1s default, a 1.9s span holding 0.9s of loud speech at its
    end measured 0.00s — one whole window (silent) plus a remainder that was
    discarded outright — and the span was skipped before decoding. Callers
    gating short spans pass a fraction of a second; the recording-level guard
    keeps 1s, where the coarse window is the point (brief noise must not count
    as speech). The trailing partial window is now measured too, at its real
    duration, instead of being dropped.
    """
    if waveform.size == 0:
        return 0.0
    window = max(1, int(round(sample_rate * window_seconds)))
    if waveform.size < window:
        rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
        return waveform.size / sample_rate if rms > rms_threshold else 0.0
    whole = waveform.size // window
    frames = waveform[: whole * window].reshape(whole, window)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    active = float(np.sum(rms > rms_threshold)) * (window / sample_rate)
    rest = waveform[whole * window :]
    if rest.size:
        rest_rms = float(np.sqrt(np.mean(np.square(rest, dtype=np.float64))))
        if rest_rms > rms_threshold:
            active += rest.size / sample_rate
    return active
