"""Recordings that keep the microphone and the meeting tab apart (extension v0.3.14+).

The recorder's left channel is the capturing user's microphone and the right one is
what the call plays in the tab. Usually they hold different people and belong
together. Not when the capturing laptop sits in a room whose microphone in the call
is another laptop: the tab then brings the room back a fraction of a second after
the microphone heard it, and summed they read as every person speaking twice (on
NOTSOFAR, two devices of one room 0.3 s apart: voices heard 3-7 -> 6-8, turns +74%,
cpWER 38.5 -> 91.8). Nobody has to know which laptop is which: the delay gives it
away, and the tab, which holds the whole call, is then the recording to listen to.

No GPU; fully unit-testable.
"""
from __future__ import annotations

import logging
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio_io import SAMPLE_RATE

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 8
# How far behind the microphone the tab can be: a call is never faster than this, and
# a laptop's own speakers echo the other way round (the microphone hears the tab
# late), which must not count: there the tab lacks the capturing user's voice.
MIN_LAG_SECONDS = 0.03
MAX_LAG_SECONDS = 2.0
# Below this neither channel is speaking for the comparison.
MIN_WINDOW_RMS = 0.003
# Peak of the whitened cross-correlation over its typical level, for a window to
# count as the same sound on both channels. On public rooms: 46 at the median for a
# room heard twice, also through noise suppression, levelling and 24 kbps Opus, and
# 7 for unrelated channels, 2% of whose windows pass by chance.
MIN_PEAK_RATIO = 12.0
# A call's delay holds from one window to the next, a chance peak does not.
SAME_LAG_SECONDS = 0.05
# Share of the comparable windows, and how many at least, that make it a room heard
# twice: a few chance matches must not silence a microphone.
MIN_MATCHED_SHARE = 0.25
MIN_MATCHED_WINDOWS = 5
# With the room on the tab, the microphone still counts where the tab has nothing:
# someone next to the laptop the other one did not reach.
TAB_QUIET_DB = 30.0


@dataclass
class ChannelMatch:
    """How the two channels of a recording relate."""
    windows: int
    matched: int
    lag_seconds: float | None

    @property
    def heard_twice(self) -> bool:
        return (
            self.matched >= MIN_MATCHED_WINDOWS
            and self.matched / max(self.windows, 1) >= MIN_MATCHED_SHARE
        )


def load_channels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """The recording's two channels at 16 kHz."""
    done = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "2", "-ar", str(SAMPLE_RATE),
         "-f", "f32le", "-"],
        check=True, capture_output=True,
    )
    both = np.frombuffer(done.stdout, dtype=np.float32)
    return both[0::2], both[1::2]


def _peak_ratio(mic: np.ndarray, tab: np.ndarray, sample_rate: int) -> tuple[float, float]:
    """Whitened cross-correlation of a stretch of microphone with the tab that starts
    at the same instant and runs MAX_LAG_SECONDS longer: (peak / typical level, lag)."""
    size = len(mic) + len(tab)
    spectrum = np.conj(np.fft.rfft(mic, size)) * np.fft.rfft(tab, size)
    magnitude = np.abs(spectrum)
    spectrum = spectrum / np.maximum(magnitude, 1e-12)
    frequencies = np.fft.rfftfreq(size, 1 / sample_rate)
    spectrum[(frequencies < 200) | (frequencies > 3400)] = 0
    correlation = np.abs(np.fft.irfft(spectrum, size))
    low, high = int(MIN_LAG_SECONDS * sample_rate), int(MAX_LAG_SECONDS * sample_rate)
    lags = correlation[low:high]
    typical = float(np.median(lags)) or 1e-12
    best = int(np.argmax(lags))
    return float(lags[best]) / typical, (best + low) / sample_rate


def match_channels(mic: np.ndarray, tab: np.ndarray, sample_rate: int = SAMPLE_RATE) -> ChannelMatch:
    """Compare the channels window by window: does the tab repeat the microphone?"""
    length = WINDOW_SECONDS * sample_rate
    extra = int(MAX_LAG_SECONDS * sample_rate)
    windows = 0
    found: list[tuple[int, float]] = []  # (window, lag) of each peak
    for start in range(0, min(len(mic), len(tab)) - length - extra, length):
        mic_part = mic[start:start + length]
        tab_part = tab[start:start + length + extra]
        if min(np.sqrt(np.mean(np.square(mic_part))), np.sqrt(np.mean(np.square(tab_part)))) < MIN_WINDOW_RMS:
            continue
        windows += 1
        ratio, lag = _peak_ratio(mic_part, tab_part, sample_rate)
        if ratio >= MIN_PEAK_RATIO:
            found.append((start // length, lag))
    lags = [
        lag for index, (window, lag) in enumerate(found)
        if any(
            abs(found[other][0] - window) <= 2 and abs(found[other][1] - lag) <= SAME_LAG_SECONDS
            for other in (index - 1, index + 1) if 0 <= other < len(found)
        )
    ]
    return ChannelMatch(windows, len(lags), float(np.median(lags)) if lags else None)


def _rms_per_second(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    whole = len(signal) // sample_rate
    frames = signal[: whole * sample_rate].reshape(whole, sample_rate)
    return np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))


def _tab_then_mic(mic: np.ndarray, tab: np.ndarray, sample_rate: int) -> np.ndarray:
    """The tab, and the microphone in the seconds the tab has nothing."""
    rms = _rms_per_second(tab, sample_rate)
    audible = rms[rms > 1e-4]
    quiet_below = (np.percentile(audible, 90) if audible.size else 0.0) * 10 ** (-TAB_QUIET_DB / 20)
    gain = np.zeros(len(tab), dtype=np.float32)
    for second in np.flatnonzero(rms <= quiet_below):
        gain[second * sample_rate:(second + 1) * sample_rate] = 1.0
    ramp = sample_rate // 20
    gain = np.convolve(gain, np.full(ramp, 1 / ramp, dtype=np.float32), mode="same")
    return tab + gain * mic


def mix_for_listening(mic: np.ndarray, tab: np.ndarray, sample_rate: int = SAMPLE_RATE) -> tuple[np.ndarray, ChannelMatch]:
    """One signal for everything after the recording: both channels summed, which is
    what the extension used to record, unless the tab repeats the microphone."""
    size = min(len(mic), len(tab))
    mic, tab = mic[:size], tab[:size]
    match = match_channels(mic, tab, sample_rate)
    mixed = _tab_then_mic(mic, tab, sample_rate) if match.heard_twice else mic + tab
    return np.clip(mixed, -1.0, 1.0), match


def write_wav(path: Path, signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes((np.clip(signal, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
