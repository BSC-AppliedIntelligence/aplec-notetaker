"""One-time Google Calendar OAuth flow.

Usage:
    python -m fred_live_worker.gcal_auth --client-secret path/to/client_secret.json

Prereq (manual, once): create an OAuth client of type "Desktop app" in Google
Cloud Console (APIs & Services -> Credentials) with the Google Calendar API
enabled, and download its JSON.

Runs the local-server consent flow, then stores the refresh-token credentials
at data/credentials/<email>.json (0600). Point FRED_LIVE_GCAL_CREDENTIALS there.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import get_settings

# calendar.events (not .readonly): the autojoin watcher PATCHes events to add the
# bot's Google account as an attendee — an invited signed-in bot joins without
# Google's "possibly risky" label.
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-secret", required=True, help="OAuth client JSON from Google Cloud Console")
    parser.add_argument(
        "--label", default="primary",
        help="Short label for the token filename (data/credentials/gcal_token_<label>.json); "
        "use one per connected calendar in multi-user setups.",
    )
    parser.add_argument(
        "--port", type=int, default=18099,
        help="Fixed local port for the OAuth redirect (default 18099). When running "
        "on a remote host, tunnel it first: ssh -L 18099:localhost:18099 <host>",
    )
    args = parser.parse_args()

    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    flow = InstalledAppFlow.from_client_secrets_file(args.client_secret, SCOPES)
    print(
        f"\nIf this machine is remote, open an SSH tunnel from your laptop first:\n"
        f"    ssh -L {args.port}:localhost:{args.port} <user>@<this-host>\n"
        f"then open the URL below in YOUR browser and complete the consent flow.\n"
    )
    creds = flow.run_local_server(port=args.port, open_browser=False)

    # Persist the token IMMEDIATELY — never let a post-consent step lose it.
    settings = get_settings()
    out_dir = settings.data_dir / "credentials"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gcal_token_{args.label}.json"
    out_path.write_text(creds.to_json(), encoding="utf-8")
    out_path.chmod(0o600)

    # Best-effort: identify the connected account (needs no extra scope —
    # calendars().get on primary is covered by calendar.events).
    account = "unknown"
    try:
        service = build("calendar", "v3", credentials=creds, cache_discovery=False)
        account = service.calendars().get(calendarId="primary").execute().get("id", "unknown")
    except Exception:
        pass

    print(f"\n✅ Connected calendar account: {account}")
    print(f"Credentials stored at {out_path}")
    print(f"Set FRED_LIVE_GCAL_CREDENTIALS={out_path} in .env.worker")


if __name__ == "__main__":
    main()
