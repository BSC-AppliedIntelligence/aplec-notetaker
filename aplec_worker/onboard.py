"""One-step coworker onboarding (run on the host where Vexa runs):

    python -m aplec_worker.onboard --email ana@example.com --name "Ana"
    python -m aplec_worker.onboard --email ana@example.com --remove

One run does everything a new user needs on the server side:

1. find-or-create the Vexa user (admin API, port 8057) — idempotent;
2. mint a user-scoped API key (scope=bot);
3. register that user's webhook (gateway ``PUT /user/webhook``) pointing at
   this worker, with the shared HMAC secret — without this, Vexa records the
   user's meetings but never tells us they finished;
4. store the key in ``data/credentials/vexa_keys.json`` (0600) so the
   autojoin watcher launches THEIR calendar's bots as THEM;
5. add the user to ``data/users.json`` — the web app and the capture-token
   check read it per request, so login works immediately, no restarts.

What it cannot do: put the person on the Google OAuth consent screen (add
them as a test user in the Cloud console, or publish the app), or do their
personal setup (login, connect calendar, extension, voiceprint) — the
printed checklist covers those.

``--remove`` offboards: drops the allowlist entry and the stored key (their
session, capture token and autojoin launches all stop on the next request).
The Vexa user and their historical meetings are left intact.

Re-running for an existing user is safe and mints a FRESH key (the previous
one stays valid in Vexa; rotate/revoke there if needed).
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import httpx

from .config import get_settings
from .users import add_file_user, remove_file_user, remove_vexa_key, store_vexa_key

logger = logging.getLogger(__name__)

ADMIN_API_URL = "http://127.0.0.1:8057"
DEFAULT_MAX_BOTS = 3  # mirrors the operator account


def resolve_admin_token(explicit: str | None = None) -> str:
    """VEXA_ADMIN_TOKEN env / --admin-token, else ADMIN_TOKEN from the
    vendored Vexa .env (single-host deployment: same box, same operator)."""
    import os

    if explicit:
        return explicit
    if os.environ.get("VEXA_ADMIN_TOKEN"):
        return os.environ["VEXA_ADMIN_TOKEN"]
    env_path = Path(__file__).resolve().parent.parent / "vendor" / "vexa" / ".env"
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^ADMIN_TOKEN=(.+)$", line.strip())
            if m:
                return m.group(1).strip()
    except OSError:
        pass
    raise SystemExit(
        "No admin token: set VEXA_ADMIN_TOKEN, pass --admin-token, "
        f"or keep ADMIN_TOKEN readable in {env_path}"
    )


def webhook_url(settings) -> str:
    token = settings.auth_token.get_secret_value() if settings.auth_token else ""
    if not token:
        raise SystemExit("APLEC_AUTH_TOKEN is not set; webhook cannot be registered")
    return f"http://host.docker.internal:18088/internal/vexa/meeting-completed?token={token}"


def onboard_user(
    settings,
    email: str,
    *,
    name: str | None = None,
    admin: bool = False,
    max_bots: int = DEFAULT_MAX_BOTS,
    admin_token: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> dict:
    """Returns {user_id, api_key, created}. Network errors raise."""
    email = email.strip().lower()
    admin_token = resolve_admin_token(admin_token)
    gateway = (settings.vexa_api_base_url or "http://localhost:8056").rstrip("/")

    with httpx.Client(transport=transport, timeout=30) as client:
        # 1. find-or-create the Vexa user (201 created / 200 found).
        resp = client.post(
            f"{ADMIN_API_URL}/admin/users",
            headers={"X-Admin-API-Key": admin_token},
            json={"email": email, "name": name, "max_concurrent_bots": max_bots},
        )
        resp.raise_for_status()
        user = resp.json()
        user_id, created = int(user["id"]), resp.status_code == 201

        # 2. mint a user-scoped API key.
        resp = client.post(
            f"{ADMIN_API_URL}/admin/users/{user_id}/tokens",
            headers={"X-Admin-API-Key": admin_token},
            params={"scope": "bot", "name": "aplec-notetaker"},
        )
        resp.raise_for_status()
        api_key = resp.json()["token"]

        # 3. register the per-user webhook WITH THE FRESH KEY (X-API-Key auth):
        # meeting.completed for this user's meetings must reach this worker.
        body = {"webhook_url": webhook_url(settings)}
        if settings.webhook_secret:
            body["webhook_secret"] = settings.webhook_secret.get_secret_value()
        resp = client.put(
            f"{gateway}/user/webhook", headers={"X-API-Key": api_key}, json=body
        )
        resp.raise_for_status()

    # 4 + 5. persist AFTER every network step succeeded — a half-onboarded
    # user (key stored, webhook missing) would record meetings that silently
    # never reach the pipeline.
    store_vexa_key(settings.data_dir, email, api_key)
    add_file_user(settings.data_dir, email, user_id, admin=admin)
    return {"user_id": user_id, "api_key": api_key, "created": created}


def offboard_user(settings, email: str) -> dict:
    email = email.strip().lower()
    return {
        "allowlist_removed": remove_file_user(settings.data_dir, email),
        "key_removed": remove_vexa_key(settings.data_dir, email),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--email", required=True)
    parser.add_argument("--name", default=None, help="Display name for the Vexa user")
    parser.add_argument("--admin", action="store_true", help="Admin in the web app (sees all meetings)")
    parser.add_argument("--max-bots", type=int, default=DEFAULT_MAX_BOTS)
    parser.add_argument("--admin-token", default=None, help="Vexa admin API token override")
    parser.add_argument("--remove", action="store_true", help="Offboard instead of onboard")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()

    if args.remove:
        result = offboard_user(settings, args.email)
        print(f"Offboarded {args.email}: allowlist_removed={result['allowlist_removed']} "
              f"key_removed={result['key_removed']}")
        print("Their session, capture token and autojoin launches stop on the next request.")
        return

    try:
        result = onboard_user(
            settings, args.email, name=args.name, admin=args.admin,
            max_bots=args.max_bots, admin_token=args.admin_token,
        )
    except httpx.HTTPStatusError as exc:
        sys.exit(f"Onboarding failed: {exc.response.status_code} {exc.response.text[:300]}")
    except httpx.HTTPError as exc:
        sys.exit(f"Onboarding failed (is the Vexa stack up?): {exc}")

    verb = "created" if result["created"] else "already existed"
    print(f"""
Onboarded {args.email} (Vexa user {result['user_id']}, {verb}){' as ADMIN' if args.admin else ''}.
Server side is done: webhook registered, key stored, allowlist updated (no restarts needed).

Remaining steps:
  1. Google Cloud console: add {args.email} as a test user on the OAuth consent
     screen (or publish the app) — without this, Google blocks their login.
  2. Tell them to open your Aplec site (APLEC_WEB_BASE_URL),
     sign in with Google, and in Settings:
       - connect their Google Calendar (enables auto-join for their meetings),
       - install the browser extension + paste their token,
       - enroll their voiceprint (or you send an invite from the Team page).
""")


if __name__ == "__main__":
    main()
