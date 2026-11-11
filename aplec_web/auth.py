from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from fastapi import Request
from fastapi.responses import RedirectResponse
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from .config import WebSettings, WebUser

logger = logging.getLogger(__name__)

SESSION_COOKIE = "aplec_session"
_SALT = "aplec-notetaker-web"


@dataclass(frozen=True)
class SessionUser:
    email: str
    name: str
    vexa_user_id: int
    is_admin: bool


def _serializer(settings: WebSettings, salt: str = _SALT) -> URLSafeTimedSerializer:
    if not settings.session_secret:
        raise RuntimeError(
            "APLEC_WEB_SESSION_SECRET is not set; generate one with "
            "`openssl rand -hex 32` and add it to .env.worker"
        )
    return URLSafeTimedSerializer(settings.session_secret, salt=salt)


def issue_session(settings: WebSettings, response, user: WebUser, name: str | None = None) -> None:
    token = _serializer(settings).dumps(
        {"email": user.email, "name": name or user.email.split("@")[0]}
    )
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=settings.session_max_age_seconds,
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
    )


def clear_session(response) -> None:
    response.delete_cookie(SESSION_COOKIE)


def current_user(settings: WebSettings, request: Request) -> SessionUser | None:
    """Resolve the session cookie against the CURRENT allowlist — removing an
    email from APLEC_WEB_USERS revokes access on the next request."""
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    try:
        data = _serializer(settings).loads(token, max_age=settings.session_max_age_seconds)
    except (BadSignature, SignatureExpired):
        return None
    email = str(data.get("email", "")).lower()
    mapped = settings.user_map().get(email)
    if mapped is None:
        return None
    return SessionUser(
        email=email,
        name=str(data.get("name") or email.split("@")[0]),
        vexa_user_id=mapped.vexa_user_id,
        is_admin=mapped.is_admin,
    )


def login_redirect(request: Request) -> RedirectResponse:
    return RedirectResponse(url="/login", status_code=303)


# --- Google OAuth (activates once GOOGLE_OAUTH_CLIENT_ID/SECRET are set) ---

CALENDAR_SCOPE = "https://www.googleapis.com/auth/calendar.events"


def build_oauth(settings: WebSettings):
    """Return an authlib OAuth registry with two clients on the same Web
    credential (and the same registered redirect URI): `google` for login and
    `google_cal` for the calendar-connect flow, or None when the Web client
    hasn't been configured yet."""
    if not settings.google_oauth_configured:
        return None
    from authlib.integrations.starlette_client import OAuth

    oauth = OAuth()
    common = dict(
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    )
    oauth.register(name="google", client_kwargs={"scope": "openid email profile"}, **common)
    oauth.register(
        name="google_cal",
        client_kwargs={"scope": f"openid email {CALENDAR_SCOPE}"},
        **common,
    )
    return oauth


def calendar_token_path(settings: WebSettings, email: str):
    """Per-user calendar refresh-token file, consumed by the autojoin watcher."""
    safe = "".join(c if c.isalnum() or c in "@.-_" else "_" for c in email.lower())
    return settings.data_dir / "credentials" / f"gcal_token_{safe}.json"


def store_calendar_token(settings: WebSettings, email: str, token: dict) -> tuple[str, str]:
    """Persist an authlib token response as a google-auth credentials file.

    Returns (path, connected_google_account). Requires a refresh_token — the
    connect flow always requests access_type=offline & prompt=consent.
    """
    if not token.get("refresh_token"):
        raise ValueError("Google did not return a refresh token; retry the connect flow")
    from google.oauth2.credentials import Credentials

    creds = Credentials(
        token=token.get("access_token"),
        refresh_token=token["refresh_token"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        scopes=[CALENDAR_SCOPE],
    )
    account = str((token.get("userinfo") or {}).get("email") or "unknown")
    path = calendar_token_path(settings, email)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(creds.to_json())
    payload["connected_account"] = account
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)
    return str(path), account


def read_calendar_connection(settings: WebSettings, email: str) -> str | None:
    """The connected Google account for this user, or None if not connected."""
    path = calendar_token_path(settings, email)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return str(data.get("connected_account") or "connected")
    except (OSError, json.JSONDecodeError):
        return None
