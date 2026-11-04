"""Short-span spoken language ID (ECAPA-TDNN on VoxLingua107, via SpeechBrain).

Whisper's language detection is near chance on sub-8s code-switched chunks
(fast turn-taking produced sv/pl/pt hallucinations in a pure EN/TR meeting).
The dedicated ECAPA classifier is the practical SOTA for short utterances
(~6.5% err at 2.5s on VoxLingua107) — but 107-way top-1 is STILL unreliable
under ~2s, so the pipeline never uses it raw:

1. spans >= evidence_seconds establish the meeting's language INVENTORY;
2. shorter spans get ECAPA probabilities renormalized over that inventory;
3. a stickiness prior (same speaker keeps their language; conversations don't
   flip language every two seconds) breaks noise-level ties. Confident
   detections dominate the prior by construction (log-space bonuses).
"""
from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Log-space stickiness bonuses (e^2 ~ 7.4x, e^1 ~ 2.7x). Validated on meeting
# 17: strong enough to fix noise-level 1.7s chunks, too weak to flip a
# confident detection (p>=0.9 wins by >4 nats over any prior).
SAME_SPEAKER_BONUS = 2.0
PREV_LANGUAGE_BONUS = 1.0
_PROB_FLOOR = 1e-8


@dataclass
class LangSpan:
    start: float
    end: float
    speaker: str | None
    language: str | None

    @property
    def duration(self) -> float:
        return self.end - self.start


def language_inventory(
    spans: list[LangSpan],
    evidence_seconds: float,
    min_span_seconds: float = 4.0,
) -> list[str]:
    """Languages evidenced by CUMULATIVE whisper-detected duration:
    total >= evidence_seconds, counting only spans >= min_span_seconds.
    Ordered by total evidenced duration.

    A single long span still qualifies (its own duration >= evidence_seconds),
    but so does genuine code-switching that never holds one language for that
    long: field failure was English spoken in 7.9s + 6.9s spans in a Portuguese
    meeting — each just under the 8s bar, 14.8s together — which left 'en'
    off-inventory, and the ECAPA repair (accent-biased: Brazilian English reads
    as 'pt') made Whisper re-transcribe = TRANSLATE it into Portuguese. The
    min_span floor keeps hallucination confetti out: interjection-length spans
    have near-chance LID (observed sv/pl on 1-2s Turkish), and summing them
    would let repeated noise qualify."""
    totals: dict[str, float] = {}
    for span in spans:
        if span.language and span.duration >= min_span_seconds:
            totals[span.language] = totals.get(span.language, 0.0) + span.duration
    return sorted(
        (lang for lang, total in totals.items() if total >= evidence_seconds),
        key=totals.get,
        reverse=True,
    )


def confirm_out_of_inventory(
    whisper_lang: str | None,
    inventory: list[str],
    probs: dict[str, float],
    duration: float,
    min_prob: float = 0.85,
    min_seconds: float = 3.0,
) -> bool:
    """Code-switching escape valve: is Whisper's OUT-OF-inventory detection a
    genuine aside rather than a hallucination? True iff ECAPA independently
    agrees on the same language, with confidence, on a long-enough span.

    Two independent modalities must agree: Whisper reads words (it hallucinated
    sv/pl on Turkish), ECAPA reads timbre (it called Brazilian-accented English
    'pt' at 0.986). Neither alone is trustworthy for admitting a NEW language,
    but their agreement is. Gates from real failures: the 'Jocksa och Kejme'
    hallucination scored sv=0.88 on 1.7s (blocked by duration); most garbage
    scored <=0.67 (blocked by prob)."""
    if not whisper_lang or whisper_lang in inventory:
        return False
    if duration < min_seconds:
        return False
    return probs.get(whisper_lang, 0.0) >= min_prob


def pick_language(
    probs: dict[str, float],
    inventory: list[str],
    prev_same_speaker: str | None,
    prev_any: str | None,
) -> str | None:
    """Constrained, stickiness-smoothed language decision for one short span.

    score(L) = log P_ecapa(L) + 2.0*[L == prev same-speaker lang] + 1.0*[L == prev lang]
    """
    if not inventory:
        return None
    best_lang, best_score = None, -math.inf
    for lang in inventory:
        score = math.log(max(probs.get(lang, 0.0), _PROB_FLOOR))
        if lang == prev_same_speaker:
            score += SAME_SPEAKER_BONUS
        if lang == prev_any:
            score += PREV_LANGUAGE_BONUS
        if score > best_score:
            best_lang, best_score = lang, score
    return best_lang


class EcapaLanguageId:
    """Lazy-loaded speechbrain/lang-id-voxlingua107-ecapa. probs() returns a
    {iso_code: probability} dict over all 107 languages."""

    def __init__(self, settings):
        self.settings = settings
        self._classifier = None
        self._lock = threading.Lock()

    def _get(self):
        with self._lock:
            if self._classifier is None:
                from speechbrain.inference.classifiers import EncoderClassifier

                logger.info("Loading ECAPA language-id model (first use)")
                self._classifier = EncoderClassifier.from_hparams(
                    source="speechbrain/lang-id-voxlingua107-ecapa",
                    savedir=str(self.settings.lid_model_dir),
                )
            return self._classifier

    def probs(self, waveform, start: float, end: float, sample_rate: int = 16000) -> dict[str, float]:
        """Language probabilities for waveform[start:end] (seconds)."""
        import torch

        clf = self._get()
        chunk = waveform[int(start * sample_rate) : int(end * sample_rate)]
        if len(chunk) < sample_rate // 4:  # <0.25s: not classifiable
            return {}
        tensor = torch.from_numpy(chunk).unsqueeze(0)
        out = clf.classify_batch(tensor)[0].exp().squeeze()
        result: dict[str, float] = {}
        for idx, prob in enumerate(out.tolist()):
            label = clf.hparams.label_encoder.decode_ndim(idx)  # "tr: Turkish"
            code = label.split(":", 1)[0].strip()
            result[code] = prob
        return result
