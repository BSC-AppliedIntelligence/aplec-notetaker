"""Enroll, list, and delete voiceprints for known-speaker identification.

Usage:
  python -m aplec_worker.enroll --name "Ada Lovelace" a.wav b.m4a
  python -m aplec_worker.enroll --name "Ada Lovelace" --add --note "room" c.wav
  python -m aplec_worker.enroll --list
  python -m aplec_worker.enroll --delete "Ada Lovelace"

Any audio format ffmpeg can decode works (wav/m4a/mp3/webm/ogg). Aim for
30-60s of clean solo speech per person, ideally from the same kind of source
as real meetings (laptop mic). Multiple files are averaged into one print.
--name replaces any print of that name; --add brings the files into the existing one.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import get_settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("enroll")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", help="Person's display name (as it should appear in transcripts)")
    parser.add_argument("samples", nargs="*", help="Audio files with this person speaking alone")
    parser.add_argument("--add", action="store_true", help="Add the samples to the existing print")
    parser.add_argument("--note", help="With --add: where the samples came from, kept in the print")
    parser.add_argument("--list", action="store_true", help="List enrolled voiceprints")
    parser.add_argument("--delete", metavar="NAME", help="Delete a person's voiceprint")
    args = parser.parse_args()

    settings = get_settings()
    from .voiceprints import VoiceprintStore, add_samples, enroll, load_backend

    store = VoiceprintStore(settings.data_dir)

    if args.list:
        prints = store.load_all()
        if not prints:
            print("No voiceprints enrolled.")
            return 0
        for p in prints:
            print(f"  {p.name:30s} {p.seconds:6.1f}s from {p.samples} sample(s)  [{p.model}]")
        return 0

    if args.delete:
        if store.delete(args.delete):
            print(f"Deleted voiceprint for {args.delete}")
            return 0
        print(f"No voiceprint found for {args.delete}")
        return 1

    if not args.name or not args.samples:
        parser.print_help()
        return 2

    paths = [Path(p) for p in args.samples]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        print(f"Missing files: {', '.join(map(str, missing))}", file=sys.stderr)
        return 1

    from .audio_io import load_waveform_16k

    backend = load_backend(settings.voiceprint_backend, settings.voiceprint_model_dir or None)
    waveforms = [load_waveform_16k(p) for p in paths]
    if args.add:
        print_ = add_samples(store, backend, args.name.strip(), waveforms, args.note)
    else:
        print_ = enroll(store, backend, args.name.strip(), waveforms)
    print(
        f"Enrolled {print_.name}: {print_.seconds:.1f}s from {print_.samples} sample(s) "
        f"using {print_.model}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
