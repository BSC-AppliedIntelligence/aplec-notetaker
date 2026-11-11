"""Known-speaker identification from enrolled voiceprints.

The DOM speaker events name whoever owns the Meet identity, which is
structurally blind inside a hybrid room: one identity, several physical
voices. Voiceprints close that gap: each team member enrolls from a few audio
samples, and at processing time every diarization cluster is embedded and
cosine-matched against the enrolled prints. The voice vote NEVER overrides a
DOM-event name; it fills unmapped clusters and is allowed to give several
clusters the same person (diarization oversplit), which the DOM vote must not.

Backends (chosen by APLEC_VOICEPRINT_BACKEND, default "auto"):
- "redimnet": ReDimNet2 via the wespeaker toolkit (2026 verification SOTA).
  Requires `wespeaker` to be installed (not on PyPI; install from the wenet
  GitHub repo). Preferred when importable.
- "ecapa": speechbrain ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb).
  speechbrain is already a dependency (the LID model), so this works with no
  new packages; solid production baseline.
Prints are tagged with the backend's model id; prints from another model are
ignored at identify time (embedding spaces are incompatible).

Store: data/voiceprints/<slug>.json, one file per person.
Enroll: python -m aplec_worker.enroll --name "Ada" sample1.wav sample2.m4a
"""
from __future__ import annotations

import contextlib
import json
import logging
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .speaker_merge import PAGE_GROUP_MIN_SECONDS, TurnGroup, same_person

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# Per-cluster audio used for identification: the N longest turns, each capped.
MAX_TURNS_PER_CLUSTER = 3
MAX_SECONDS_PER_TURN = 10.0
MIN_CLUSTER_SECONDS = 2.0
MIN_ENROLL_SECONDS = 8.0
# What new samples weigh in a print they join.
ADDED_SHARE = 1 / 3
# A turn this long can be compared with two prints on its own.
MIN_TURN_CHECK_SECONDS = 1.5
# The most this process may hold on the GPU. Whisper and Nemotron run in processes of
# their own; pyannote and embeddings stay under 6 GB on an hour of audio.
MAX_GPU_BYTES = 30 * 1024**3
# Shortest turn worth comparing with every print, when a wrong answer can move it.
# On a real room a second recording confirmed 45 of the 46 s moved in turns of 1-2 s.
MIN_TURN_SWEEP_SECONDS = 1.0


def _slug(name: str) -> str:
    folded = re.sub(r"[^\w]+", "-", name.strip().lower()).strip("-")
    return folded or "unnamed"


@dataclass
class Voiceprint:
    name: str
    embedding: np.ndarray  # l2-normalized
    model: str
    samples: int
    seconds: float
    # Consent/provenance metadata (GDPR): who this print belongs to (account
    # email), when consent was recorded, which notice version was shown, and
    # how the audio arrived ("self-recorded" | "upload" | "cli").
    metadata: dict | None = None


class EmbeddingBackend:
    """Interface: embed 16k mono float waveforms into l2-normalized vectors."""

    model_id = "abstract"

    def embed(self, waveform: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


DEFAULT_REDIMNET_DIR = str(
    Path(__file__).resolve().parent.parent / "models" / "wespeaker__voxceleb-redimnet2-B6-LM"
)


class RedimnetBackend(EmbeddingBackend):
    """ReDimNet2B6-LM through wespeaker (2026 SOTA speaker verification).
    The checkpoint is NOT in wespeaker's tiny built-in hub; it is downloaded
    from HF (Wespeaker/wespeaker-voxceleb-redimnet2-B6-LM: avg_model.pt +
    config.yaml) and loaded by path."""

    model_id = "wespeaker-redimnet2-b6-lm"

    def __init__(self, model_dir: str | None = None):
        import sys
        import types

        import torch
        import torchaudio

        if not hasattr(torchaudio, "set_audio_backend"):
            # Removed in torchaudio 2.x; s3prl still calls it at import time.
            # It was a no-op for years; shim it.
            torchaudio.set_audio_backend = lambda *a, **k: None
        try:
            import s3prl.nn  # noqa: F401  (imports fine on older torchaudio)
        except Exception:
            # s3prl's import walks every SSL upstream and trips on more
            # removed torchaudio APIs (sox_effects, ...). wespeaker only
            # imports two names from it, used exclusively by the optional
            # S3prlFrontend at INSTANTIATION time; ReDimNet2's TFMel frontend
            # never touches them. Pre-seed a stub so the real s3prl.nn is
            # never executed.
            stub = types.ModuleType("s3prl.nn")
            stub.Featurizer = object
            stub.S3PRLUpstream = object
            sys.modules["s3prl.nn"] = stub
            sys.modules.setdefault("s3prl", types.ModuleType("s3prl"))
        import wespeaker  # not on PyPI; install from the wenet-e2e GitHub repo

        self._model = wespeaker.load_model(model_dir or DEFAULT_REDIMNET_DIR)
        try:
            self._model.set_device("cuda")
        except Exception:
            pass
        if torch.cuda.is_available():
            # On this unified-memory host a runaway request takes the whole machine
            # down, other people's work included. Better one failed embedding.
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, MAX_GPU_BYTES / total))

    @staticmethod
    @contextlib.contextmanager
    def _cudnn_for_clips():
        """Every clip has its own length, and what cuDNN does with a new length
        depends on two process-wide switches other libraries set. wespeaker's load
        turns benchmark mode on: a search per length (10 clips took 103 s against 3 s,
        one run peaked at 58 GB). pyannote turns TF32 off, and without the search the
        default algorithm for some lengths then wants over 100 GB at once: a single
        1.8 s clip exhausted the host's memory, and the kernel killed the worker with
        it. With TF32 on and no search: 1.2 GB, 0.1 s, the same embeddings."""
        import torch

        cudnn = torch.backends.cudnn
        saved = cudnn.benchmark, cudnn.allow_tf32
        cudnn.benchmark, cudnn.allow_tf32 = False, True
        try:
            yield
        finally:
            cudnn.benchmark, cudnn.allow_tf32 = saved

    def embed(self, waveform: np.ndarray) -> np.ndarray:
        import torch

        pcm = torch.from_numpy(np.array(waveform, dtype=np.float32, copy=True)).unsqueeze(0)
        # extract_embedding_from_pcm computes features on the INPUT's device;
        # the model weights live wherever set_device put them. Align them.
        pcm = pcm.to(self._model.device)
        with self._cudnn_for_clips():
            emb = self._model.extract_embedding_from_pcm(pcm, SAMPLE_RATE)
        if emb is None:  # wespeaker's internal VAD found no speech
            raise ValueError("no speech in audio (wespeaker VAD)")
        vec = emb.detach().cpu().numpy().reshape(-1).astype(np.float64)
        return vec / (np.linalg.norm(vec) or 1.0)


class EcapaBackend(EmbeddingBackend):
    """speechbrain ECAPA-TDNN; zero new dependencies on this host."""

    model_id = "speechbrain-ecapa-voxceleb"

    def __init__(self, model_dir: str | None = None):
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        self._torch = torch
        self._model = EncoderClassifier.from_hparams(
            source=model_dir or "speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )

    def embed(self, waveform: np.ndarray) -> np.ndarray:
        wav = self._torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32)).unsqueeze(0)
        emb = self._model.encode_batch(wav)
        vec = emb.detach().cpu().numpy().reshape(-1).astype(np.float64)
        return vec / (np.linalg.norm(vec) or 1.0)


def load_backend(kind: str = "auto", model_dir: str | None = None) -> EmbeddingBackend:
    if kind in ("auto", "redimnet"):
        try:
            return RedimnetBackend(model_dir)
        except ImportError:
            if kind == "redimnet":
                raise RuntimeError(
                    "APLEC_VOICEPRINT_BACKEND=redimnet but the wespeaker package is "
                    "not installed (it is not on PyPI; install it from the wenet-e2e "
                    "GitHub repo, then re-run)."
                )
            logger.info("Voiceprints: wespeaker not installed; using ECAPA backend")
    return EcapaBackend(model_dir if kind == "ecapa" else None)


class VoiceprintStore:
    def __init__(self, data_dir: Path):
        self.dir = Path(data_dir) / "voiceprints"

    def _path(self, name: str) -> Path:
        return self.dir / f"{_slug(name)}.json"

    def save(self, print_: Voiceprint) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self._path(print_.name)
        payload = {
            "name": print_.name,
            "embedding": [round(float(x), 8) for x in print_.embedding],
            "model": print_.model,
            "samples": print_.samples,
            "seconds": round(print_.seconds, 1),
            "updated_at": int(time.time()),
        }
        if print_.metadata:
            payload["metadata"] = print_.metadata
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def delete(self, name: str) -> bool:
        path = self._path(name)
        if path.is_file():
            path.unlink()
            return True
        return False

    def delete_slug(self, slug: str) -> bool:
        if not re.fullmatch(r"[\w-]+", slug):
            return False
        path = self.dir / f"{slug}.json"
        if path.is_file():
            path.unlink()
            return True
        return False

    def set_account(self, slug: str, email: str | None, by: str) -> dict | None:
        """Link a print to an Aplec account (its email), or unlink it (None).
        Prints enrolled with the operator CLI before the consent flow carry no
        email; this is how an admin attaches one, so the person can be shared
        meetings automatically. Returns the print's metadata, or None when no
        such print exists."""
        if not re.fullmatch(r"[\w-]+", slug):
            return None
        path = self.dir / f"{slug}.json"
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        if email:
            metadata["email"] = email.strip().lower()[:120]
            metadata["linked_by"] = by.strip().lower()[:120]
            metadata["linked_at"] = int(time.time())
        else:
            metadata.pop("email", None)
            metadata.pop("linked_by", None)
            metadata.pop("linked_at", None)
        data["metadata"] = metadata
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return metadata

    def slug_for(self, name: str) -> str:
        return _slug(name)

    def load_all(self, model: str | None = None) -> list[Voiceprint]:
        prints: list[Voiceprint] = []
        if not self.dir.is_dir():
            return prints
        for path in sorted(self.dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                vp = Voiceprint(
                    name=str(data["name"]),
                    embedding=np.asarray(data["embedding"], dtype=np.float64),
                    model=str(data.get("model") or "?"),
                    samples=int(data.get("samples") or 1),
                    seconds=float(data.get("seconds") or 0),
                    metadata=data.get("metadata") if isinstance(data.get("metadata"), dict) else None,
                )
            except (KeyError, ValueError, json.JSONDecodeError):
                logger.warning("Voiceprints: unreadable print %s; skipping", path.name)
                continue
            if model and vp.model != model:
                logger.warning(
                    "Voiceprints: %s was enrolled with %s but the active backend is %s; "
                    "re-enroll to use it", vp.name, vp.model, model,
                )
                continue
            prints.append(vp)
        return prints


def mean_embedding(vectors: list[np.ndarray]) -> np.ndarray:
    mean = np.mean(np.stack(vectors), axis=0)
    return mean / (np.linalg.norm(mean) or 1.0)


def _sample_vectors(backend: EmbeddingBackend, waveforms: list[np.ndarray]) -> tuple[list[np.ndarray], float]:
    """One embedding per ~20s chunk of each sample, so one rambling file doesn't
    dominate variance, and the samples' total seconds."""
    vectors: list[np.ndarray] = []
    seconds = 0.0
    for wav in waveforms:
        seconds += len(wav) / SAMPLE_RATE
        step = 20 * SAMPLE_RATE
        chunks = [wav[i:i + step] for i in range(0, len(wav), step)]
        for chunk in chunks:
            if len(chunk) < int(1.5 * SAMPLE_RATE):
                continue
            vectors.append(backend.embed(chunk))
    if not vectors:
        raise ValueError("No usable audio in the provided samples (need >= 1.5s chunks)")
    return vectors, seconds


def enroll(store: VoiceprintStore, backend: EmbeddingBackend, name: str,
           waveforms: list[np.ndarray], metadata: dict | None = None) -> Voiceprint:
    """Average one embedding per provided sample, replacing any print of that name."""
    vectors, seconds = _sample_vectors(backend, waveforms)
    if seconds < MIN_ENROLL_SECONDS:
        logger.warning(
            "Voiceprints: only %.1fs of audio for %s; identification quality improves "
            "with %d+ seconds", seconds, name, int(MIN_ENROLL_SECONDS),
        )
    print_ = Voiceprint(
        name=name, embedding=mean_embedding(vectors), model=backend.model_id,
        samples=len(waveforms), seconds=seconds, metadata=metadata,
    )
    store.save(print_)
    logger.info(
        "Voiceprints: enrolled %s (%d sample(s), %.1fs, %d chunk embeddings, model=%s)",
        name, len(waveforms), seconds, len(vectors), backend.model_id,
    )
    return print_


def add_samples(store: VoiceprintStore, backend: EmbeddingBackend, name: str,
                waveforms: list[np.ndarray], note: str | None = None) -> Voiceprint:
    """Bring new samples into an existing print, ADDED_SHARE of it, and record where
    they came from. A print from a quiet sample recognized 51-66% of its owner's
    speech in real rooms, and a third from validated room speech moved three of them
    55 -> 76%, 69 -> 78 and 57 -> 66, with the clips of the meeting under test left
    out. More is not better: at a half (80, 80, 71) the worst impostor crossed the
    gate and 18 minutes of a page-named speaker took the second one's name at 0.551;
    each clip weighing as much as the old print lost 271 s of that second speaker."""
    existing = next((p for p in store.load_all(backend.model_id) if p.name == name), None)
    if existing is None:
        raise ValueError(f"No voiceprint for {name} with model {backend.model_id}")
    vectors, seconds = _sample_vectors(backend, waveforms)
    metadata = dict(existing.metadata or {})
    metadata["added_samples"] = [*metadata.get("added_samples", []), {
        "at": int(time.time()), "samples": len(waveforms), "seconds": round(seconds, 1), "note": note,
    }]
    mixed = (1 - ADDED_SHARE) * existing.embedding + ADDED_SHARE * mean_embedding(vectors)
    print_ = Voiceprint(
        name=name, embedding=mixed / (np.linalg.norm(mixed) or 1.0),
        model=existing.model, samples=existing.samples + len(waveforms),
        seconds=existing.seconds + seconds, metadata=metadata,
    )
    store.save(print_)
    logger.info("Voiceprints: added %d sample(s), %.1fs, to %s", len(waveforms), seconds, name)
    return print_


def cluster_representative_audio(waveform: np.ndarray, turns: list, label: str) -> np.ndarray | None:
    """Concatenate the cluster's longest turns (each capped) as ID audio."""
    own = sorted(
        (t for t in turns if t.speaker == label),
        key=lambda t: (t.end - t.start), reverse=True,
    )[:MAX_TURNS_PER_CLUSTER]
    pieces = []
    total = 0.0
    for t in own:
        a = int(t.start * SAMPLE_RATE)
        b = int(min(t.end, t.start + MAX_SECONDS_PER_TURN) * SAMPLE_RATE)
        piece = waveform[a:b]
        if piece.size:
            pieces.append(piece)
            total += piece.size / SAMPLE_RATE
    if not pieces or total < MIN_CLUSTER_SECONDS:
        return None
    return np.concatenate(pieces)


def identify_clusters(
    waveform: np.ndarray,
    turns: list,
    existing_mapping: dict[str, str],
    prints: list[Voiceprint],
    backend: EmbeddingBackend,
    threshold: float,
    margin: float,
    override_dom: bool = True,
    accepted: list | None = None,
    on_page=None,
    off_page_threshold: float | None = None,
) -> dict[str, str]:
    """Voice vote per cluster. Returns the label->name assignments to apply.
    Unlike the DOM vote, several labels may resolve to the same person:
    diarization oversplit is exactly what this repairs.

    With override_dom (the default, a decision from field use): a
    gate-passing voice match REPLACES a conflicting DOM name. Rationale: in a
    hybrid room every voice arrives through the mic owner's Meet identity, so
    DOM attribution is systematically wrong for everyone else in the room
    (confirmed on a real meeting: the room mic's identity blanketed 85% of
    the meeting and claimed another participant's cluster).

    An unenrolled speaker CAN take a name: in one meeting a guest's 12
    minutes scored 0.607 and 0.621 against a colleague's print, every turn of it
    closer to that print than to any other, while the colleague was not there.
    Voice alone cannot tell that from a real colleague far from the microphone
    (0.573 and 0.606 in another one), but the page can: on_page(print) says whether
    the meeting vouches for that person (a tile, or the meeting is theirs), and
    one it does not needs off_page_threshold.

    accepted: when given, receives one entry per gate-passing match (DOM
    agreement included), with the print's account email, for automatic
    sharing: {"speaker", "name", "email", "score"}."""
    if not prints:
        return {}
    additions: dict[str, str] = {}
    labels = sorted({t.speaker for t in turns})
    for label in labels:
        audio = cluster_representative_audio(waveform, turns, label)
        if audio is None:
            continue
        try:
            emb = backend.embed(audio)
        except Exception:
            logger.exception("Voiceprints: embedding failed for %s", label)
            continue
        scored = sorted(
            ((float(np.dot(emb, p.embedding)), i) for i, p in enumerate(prints)), reverse=True
        )
        best_score, best_index = scored[0]
        best_name = prints[best_index].name
        runner = scored[1][0] if len(scored) > 1 else -1.0
        gate, absent = threshold, ""
        if off_page_threshold is not None and on_page is not None and not on_page(prints[best_index]):
            gate, absent = max(threshold, off_page_threshold), " (no tile)"
        passed = best_score >= gate and (best_score - runner) >= margin
        if passed and accepted is not None:
            accepted.append({
                "speaker": label,
                "name": best_name,
                "email": (prints[best_index].metadata or {}).get("email") or None,
                "score": round(best_score, 3),
            })
        dom_name = existing_mapping.get(label)
        if not passed:
            outcome = f"kept DOM name {dom_name}" if dom_name else "unmapped"
        elif dom_name is None:
            additions[label] = best_name
            outcome = best_name
        elif dom_name == best_name:
            outcome = f"agrees with DOM ({best_name})"
        elif override_dom:
            additions[label] = best_name
            outcome = f"OVERRIDES DOM name {dom_name} (hybrid-room policy)"
        else:
            outcome = f"kept DOM name {dom_name} (voice disagrees: {best_name})"
        logger.info(
            "Voice ID %s: best=%s %.3f runner-up=%.3f threshold>=%.2f%s margin>=%.2f -> %s",
            label, best_name, best_score, runner, gate, absent, margin, outcome,
        )
    return additions


def _still_the_clusters(waveform, turn, mine: Voiceprint, host: Voiceprint, backend) -> bool:
    """Whether a turn of an accepted group sounds more like the cluster's owner."""
    end = min(turn.end, turn.start + MAX_SECONDS_PER_TURN)
    try:
        emb = backend.embed(waveform[int(turn.start * SAMPLE_RATE):int(end * SAMPLE_RATE)])
    except Exception:
        return False
    return float(np.dot(emb, host.embedding)) >= float(np.dot(emb, mine.embedding))


def identify_page_groups(
    waveform: np.ndarray,
    groups: list,
    mapping: dict[str, str],
    prints: list[Voiceprint],
    backend: EmbeddingBackend,
    threshold: float,
    margin: float,
) -> list[dict]:
    """The page groups (speaker_merge.page_groups) whose voice is the person whose
    tile was lit, with the gates that name a cluster. Two signals that fail in
    different ways have to agree: the indicator cross-fires and on a room's device
    lights up for every voice in the room, and a few seconds of voice alone would be
    a weak reason to take turns from a cluster. The gates judge a group's longest
    turns, so the rest of a passing group is checked turn by turn against the
    cluster's owner, when the owner has a print and the turn is long enough to judge
    (on real meetings 10 s of the 96 s checked were the owner's). Returns one entry
    per accepted group, the surest first: {"group", "name", "email", "score"}."""
    accepted: list[dict] = []
    for group in groups:
        if not any(same_person(p.name, group.identity) for p in prints):
            continue
        audio = cluster_representative_audio(waveform, group.turns, group.label)
        if audio is None:
            continue
        try:
            emb = backend.embed(audio)
        except Exception:
            logger.exception("Voiceprints: embedding failed for a page group of %s", group.label)
            continue
        scored = sorted(
            ((float(np.dot(emb, p.embedding)), i) for i, p in enumerate(prints)), reverse=True
        )
        best_score, best_index = scored[0]
        best = prints[best_index]
        runner = scored[1][0] if len(scored) > 1 else -1.0
        passed = best_score >= threshold and (best_score - runner) >= margin
        ours = passed and same_person(best.name, group.identity)
        logger.info(
            "Page voice %s: %.1fs lit by %s: best=%s %.3f runner-up=%.3f -> %s",
            group.label, group.seconds, group.identity, best.name, best_score, runner,
            f"moved to {best.name}" if ours else "stays",
        )
        if ours:
            host = next(
                (p for p in prints if same_person(p.name, mapping.get(group.label))), None
            )
            kept = [
                turn for turn in group.turns
                if host is None
                or turn.end - turn.start < MIN_TURN_CHECK_SECONDS
                or not _still_the_clusters(waveform, turn, best, host, backend)
            ]
            group = replace(group, turns=kept)
            if group.seconds < PAGE_GROUP_MIN_SECONDS:
                continue
            accepted.append({
                "group": group,
                "name": best.name,
                "email": (best.metadata or {}).get("email") or None,
                "score": round(best_score, 3),
            })
    return sorted(accepted, key=lambda entry: -entry["score"])


def identify_turns(
    waveform: np.ndarray,
    turns: list,
    mapping: dict[str, str],
    present: list[str],
    prints: list[Voiceprint],
    backend: EmbeddingBackend,
    threshold: float,
    margin: float,
) -> list[dict]:
    """The turns whose voice is someone other than their cluster's owner, turn by turn.

    A name goes to a whole cluster, on its longest turns, so what diarization mixed
    into a cluster takes that name with it. Voice alone decides here, which the page
    cannot do for two people in a room sharing one tile. It leaves clean clusters
    alone (DER 17.1 merging clusters, 17.2-17.4 per turn on AMI; 3 s on a recording
    with clean clusters) and does the work where they are not: on a real room the
    other recording of the meeting confirmed 329 of the 358 s moved. Only someone
    already in the meeting (`present`: named, or with a tile) can receive a turn: a
    second of a similar voice must not bring an absent person into it. Returns one
    entry per (person, cluster), the surest first, as identify_page_groups does."""
    matrix = np.stack([print_.embedding for print_ in prints])
    found: dict[tuple[int, str], list] = {}
    scores: dict[tuple[int, str], float] = {}
    for turn in turns:
        if turn.end - turn.start < MIN_TURN_SWEEP_SECONDS:
            continue
        end = min(turn.end, turn.start + MAX_SECONDS_PER_TURN)
        try:
            row = matrix @ backend.embed(waveform[int(turn.start * SAMPLE_RATE):int(end * SAMPLE_RATE)])
        except Exception:
            continue  # no speech in it, for wespeaker's own VAD
        best = int(np.argmax(row))
        rest = np.delete(row, best)
        passed = row[best] >= threshold and row[best] - (rest.max() if rest.size else -1.0) >= margin
        name = prints[best].name
        if not passed or same_person(name, mapping.get(turn.speaker)):
            continue
        if not any(same_person(name, other) for other in present):
            continue
        key = (best, turn.speaker)
        found.setdefault(key, []).append(turn)
        scores[key] = max(scores.get(key, 0.0), float(row[best]))
    accepted = []
    for (best, label), moved in found.items():
        group = TurnGroup(identity=prints[best].name, label=label, turns=moved)
        logger.info(
            "Voice sweep %s (%s): %d turn(s), %.1fs, are %s (best %.3f)",
            label, mapping.get(label, "unnamed"), len(moved), group.seconds,
            prints[best].name, scores[(best, label)],
        )
        accepted.append({
            "group": group,
            "name": prints[best].name,
            "email": (prints[best].metadata or {}).get("email") or None,
            "score": round(scores[(best, label)], 3),
        })
    return sorted(accepted, key=lambda entry: -entry["score"])
