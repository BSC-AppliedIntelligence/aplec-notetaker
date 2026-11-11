from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class WebUser:
    email: str
    vexa_user_id: int
    is_admin: bool = False


class WebSettings(BaseSettings):
    """Web app settings. Reads the same .env.worker so ops stay single-file."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env.worker", env_file_encoding="utf-8", extra="ignore"
    )

    host: str = Field(default="127.0.0.1", alias="FRED_LIVE_WEB_HOST")
    port: int = Field(default=18091, alias="FRED_LIVE_WEB_PORT")
    # Cookie signing key. MUST be set to a random value in .env.worker.
    session_secret: str = Field(default="", alias="FRED_LIVE_WEB_SESSION_SECRET")
    session_max_age_seconds: int = Field(default=14 * 24 * 3600, alias="FRED_LIVE_WEB_SESSION_MAX_AGE")

    # Allowlist + identity map: "email:vexa_user_id[:admin]" comma-separated.
    # Meeting ownership is by Vexa user_id (from the hook); admins see all.
    users: str = Field(default="", alias="FRED_LIVE_WEB_USERS")

    # Dev-mode login: bypasses Google OAuth for THIS email only, and only when
    # the app is bound to a loopback host (SSH-tunnel access). Remove once the
    # Google OAuth Web client exists.
    dev_user: str | None = Field(default=None, alias="FRED_LIVE_WEB_DEV_USER")

    # Google OAuth Web client (empty until created in Cloud Console).
    google_client_id: str | None = Field(default=None, alias="GOOGLE_OAUTH_CLIENT_ID")
    google_client_secret: str | None = Field(default=None, alias="GOOGLE_OAUTH_CLIENT_SECRET")
    # External base URL for the OAuth redirect (e.g. https://aplec.example.org).
    base_url: str = Field(default="http://localhost:18091", alias="FRED_LIVE_WEB_BASE_URL")

    # Chrome Web Store listing (unlisted). When set, Settings hands out the
    # store link (auto-updating installs) and demotes the zip to dev fallback.
    extension_store_url: str | None = Field(
        default=None, alias="FRED_LIVE_EXTENSION_STORE_URL"
    )
    data_dir: Path = Field(default=REPO_ROOT / "data", alias="FRED_LIVE_DATA_DIR")
    worker_url: str = Field(default="http://127.0.0.1:18088", alias="FRED_LIVE_WORKER_URL")
    worker_token: str | None = Field(default=None, alias="FRED_LIVE_AUTH_TOKEN")

    def user_map(self) -> dict[str, WebUser]:
        """Env allowlist merged with data/users.json (written by
        `python -m fred_live_worker.onboard`), file read per call so onboarding
        needs no restarts. Env wins on a conflicting email."""
        from fred_live_worker.users import merged_users

        return {
            email: WebUser(email=u.email, vexa_user_id=u.vexa_user_id, is_admin=u.is_admin)
            for email, u in merged_users(self.users, self.data_dir).items()
        }

    @property
    def dev_mode_available(self) -> bool:
        """Dev login only without OAuth configured, only for a mapped user, and
        only when bound to loopback (remote access happens over an SSH tunnel)."""
        return (
            not self.google_client_id
            and self.dev_user is not None
            and self.dev_user.strip().lower() in self.user_map()
            and self.host in ("127.0.0.1", "localhost", "::1")
        )

    @property
    def google_oauth_configured(self) -> bool:
        return bool(self.google_client_id and self.google_client_secret)

    @property
    def cookie_secure(self) -> bool:
        """Secure-flag cookies whenever the public URL is https (the Traefik
        deployment). Keyed on base_url, not the bind host: behind the proxy we
        bind plain HTTP but the browser only ever sees https, and the flag is
        what stops a session cookie from leaking over a stray http request to
        the raw host:port."""
        return self.base_url.strip().lower().startswith("https://")


@lru_cache(maxsize=1)
def get_web_settings() -> WebSettings:
    return WebSettings()
