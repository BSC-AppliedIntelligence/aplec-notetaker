"""Ask a question about a processed meeting (CLI wrapper for the /ask endpoint).

Usage:
    python -m aplec_worker.ask --meeting-id 2 "Quem vai enviar os documentos?"
"""
from __future__ import annotations

import argparse
import os

import httpx

from .config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meeting-id", required=True)
    parser.add_argument("question")
    args = parser.parse_args()

    settings = get_settings()
    base_url = os.environ.get("APLEC_WORKER_URL", f"http://127.0.0.1:{settings.app_port}")
    headers = {}
    if settings.auth_token:
        headers["X-Aplec-Token"] = settings.auth_token.get_secret_value()

    response = httpx.post(
        f"{base_url.rstrip('/')}/internal/meetings/{args.meeting_id}/ask",
        json={"question": args.question},
        headers=headers,
        timeout=settings.llm_timeout_seconds,
    )
    response.raise_for_status()
    body = response.json()
    print(f"[run {body['run']}]")
    print(body["answer"])


if __name__ == "__main__":
    main()
