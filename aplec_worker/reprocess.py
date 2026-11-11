"""Re-run the LLM step against an existing transcript without re-doing ASR.

Usage:
    python -m aplec_worker.reprocess --meeting-id 2
    python -m aplec_worker.reprocess --meeting-id 2 --run 20260415T190607Z
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from .config import get_settings
from .pipeline import MeetingProcessor
from .schemas import MeetingCompletedPayload, TranscriptResult


def _resolve_run_dir(meetings_dir: Path, meeting_id: str, run: str | None) -> Path:
    meeting_dir = meetings_dir / meeting_id
    if not meeting_dir.is_dir():
        raise SystemExit(f"No runs found for meeting {meeting_id} under {meetings_dir}")

    if run:
        run_dir = meeting_dir / run
        if not run_dir.is_dir():
            raise SystemExit(f"Run {run!r} not found under {meeting_dir}")
        return run_dir

    candidates = sorted(p for p in meeting_dir.iterdir() if p.is_dir())
    if not candidates:
        raise SystemExit(f"No run dirs under {meeting_dir}")
    return candidates[-1]


async def _amain() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meeting-id", required=True, help="Meeting ID (matches data/meetings/<id>)")
    parser.add_argument("--run", help="Specific run timestamp; defaults to the latest run dir")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    settings = get_settings()
    run_dir = _resolve_run_dir(settings.meetings_dir, args.meeting_id, args.run)

    hook_path = run_dir / "hook.json"
    transcript_path = run_dir / "transcript.json"
    if not hook_path.exists() or not transcript_path.exists():
        raise SystemExit(f"Missing hook.json or transcript.json in {run_dir}")

    payload = MeetingCompletedPayload.model_validate(json.loads(hook_path.read_text(encoding="utf-8")))
    transcript = TranscriptResult.model_validate(json.loads(transcript_path.read_text(encoding="utf-8")))

    processor = MeetingProcessor(settings)
    result = await processor.reprocess_with_transcript(payload, transcript)

    summary = result["insights"].get("summary") or ""
    print(json.dumps(
        {
            "meeting_id": str(payload.meeting.id),
            "source_run": run_dir.name,
            "quality": result.get("quality"),
            "summary_preview": summary[:300],
        },
        ensure_ascii=False,
        indent=2,
    ))


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
