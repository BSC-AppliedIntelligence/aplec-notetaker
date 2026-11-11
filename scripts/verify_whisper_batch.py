#!/usr/bin/env python3
"""Check that whisper-batch still answers exactly like whisper-cli.

whisper-batch copies whisper-cli's decoding parameters instead of calling the
CLI, so a whisper.cpp update that changes the CLI's defaults would make the two
drift apart without anything failing. Run this after rebuilding whisper.cpp
and whisper-batch (needs the GPU, takes a minute or two):

    .venv/bin/python scripts/verify_whisper_batch.py <any recording>
    .venv/bin/python scripts/verify_whisper_batch.py rec.webm --spans 20

It cuts spans of several lengths from the recording and runs each through both
engines, via the same WhisperAsrService methods the worker uses: auto
language, the language pinned, the loop-retry temperature, and language ID.
Exit status 1 if any answer differs (segment texts, language, or the language
probability whisper-cli prints). The second language tag has no whisper-cli
equivalent to compare with, so it is only reported: whether this build has it,
and on how many spans it changed the text (zero would mean it does nothing).
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aplec_worker.asr import WhisperAsrService  # noqa: E402
from aplec_worker.config import get_settings  # noqa: E402

SPAN_SECONDS = (3.0, 7.0, 15.0, 28.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("recording", type=Path)
    parser.add_argument("--spans", type=int, default=12)
    args = parser.parse_args()

    settings = get_settings()
    svc = WhisperAsrService(settings)
    if svc._batch_binary() is None:
        print("whisper-batch not found (or disabled): run scripts/build_whisper_batch.sh", file=sys.stderr)
        return 2

    checked, differences, tag_changed = 0, [], 0
    with tempfile.TemporaryDirectory(prefix="aplec-verify-") as tmp, svc.batch_session():
        batch = svc._batch
        if batch is None:
            print("whisper-batch did not start (see the log above)", file=sys.stderr)
            return 2
        wav = svc.prepare_wav(args.recording, Path(tmp) / "full.wav")
        with wave.open(str(wav)) as w:
            duration = w.getnframes() / w.getframerate()
        for i in range(args.spans):
            length = min(SPAN_SECONDS[i % len(SPAN_SECONDS)], duration)
            start = (duration - length) * i / max(1, args.spans - 1)
            piece = svc._slice_wav(wav, start, start + length, Path(tmp) / f"span{i}.wav")

            def compare(label, cli, ours):
                nonlocal checked
                checked += 1
                if cli != ours:
                    differences.append((f"{start:.1f}+{length:.0f}s {label}", cli, ours))

            reference = svc._whisper_cli_decode(piece, None, None)
            pinned = (reference.get("result") or {}).get("language")
            for label, language, temperature in (
                ("auto", None, None),
                (f"pinned {pinned}", pinned, None),
                ("temperature 0.4", None, 0.4),
            ):
                cli = reference if label == "auto" else svc._whisper_cli_decode(piece, language, temperature)
                answer = batch.request(svc._batch_decode_request(piece, language, temperature))
                ours = svc._batch_json(answer)
                compare(label, (cli["result"]["language"], [e["text"] for e in cli["transcription"]]),
                        (ours["result"]["language"], [e["text"] for e in ours["transcription"]]))
                compare(f"{label} confidence", True,
                        abs((cli["avg_logprob"] or 0.0) - (ours["avg_logprob"] or 0.0)) < 0.01)
                if label.startswith("pinned") and batch.second_language and pinned:
                    second = "es" if pinned != "es" else "en"
                    tagged = batch.request(svc._batch_decode_request(piece, pinned, None, second))
                    tag_changed += tagged.get("texts") != answer.get("texts")
            cli_lid = svc._LID_RE.search(svc._whisper_cli_detect(piece))
            our_lid = svc._LID_RE.search(batch.request({"kind": "detect", "path": str(piece)})["log"])
            compare("language ID", cli_lid and cli_lid.groups(), our_lid and our_lid.groups())
        second_language = batch.second_language

    for where, cli, ours in differences:
        print(f"DIFFERENT {where}\n  whisper-cli:   {cli}\n  whisper-batch: {ours}")
    print(f"{checked} answers compared on {args.spans} spans: "
          f"{'all identical' if not differences else f'{len(differences)} DIFFERENT'}")
    if second_language:
        print(f"second language tag: changed the text on {tag_changed} of {args.spans} spans")
    else:
        print("no second language tag in this build: rerun scripts/build_whisper_batch.sh, "
              "or patches/whisper-cpp-second-language.patch no longer applies", file=sys.stderr)
    return 1 if differences else 0


if __name__ == "__main__":
    sys.exit(main())
