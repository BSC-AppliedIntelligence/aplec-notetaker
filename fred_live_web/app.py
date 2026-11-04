from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from jinja2 import pass_context

from . import auth
from . import markdown_lite
from . import sharing
from . import signup_requests as _signup
from .config import get_web_settings
from .i18n import (
    DEFAULT_UI_LANGUAGE,
    UI_LANGUAGES,
    format_datetime_day,
    translate,
    translate_markup,
    translate_n,
)
from .languages import lang_flag, lang_label, lang_name
from fred_live_worker.user_prefs import (
    AUTOJOIN_POLICIES,
    SUMMARY_LANGUAGES,
    get_auto_share,
    get_autojoin_policy,
    get_summary_language,
    get_ui_language,
    set_auto_share,
    set_autojoin_policy,
    set_summary_language,
    set_ui_language,
)
from .meetings import MeetingIndex, can_view, visible_cards

logger = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).resolve().parent

settings = get_web_settings()
# Fail closed: SessionMiddleware protects the OAuth state/nonce (CSRF for the
# login flow). A public fallback constant would let anyone forge that state,
# so refuse to boot without a real secret rather than run on "unset".
if not settings.session_secret:
    raise RuntimeError(
        "FRED_LIVE_WEB_SESSION_SECRET is not set; refusing to start with an "
        "insecure session key. Generate one with `openssl rand -hex 32`."
    )
app = FastAPI(title="fred-live web", docs_url=None, redoc_url=None)
app.add_middleware(
    SessionMiddleware, secret_key=settings.session_secret, https_only=settings.cookie_secure
)
app.mount("/static", StaticFiles(directory=PACKAGE_DIR / "static"), name="static")

templates = Jinja2Templates(directory=PACKAGE_DIR / "templates")
templates.env.filters["markdown"] = markdown_lite.render_markdown
index = MeetingIndex(settings.data_dir)
oauth = auth.build_oauth(settings)

QUALITY_LABELS = {
    "ok": ("Processed", "ok"),
    "suspect": ("Needs review", "suspect"),
    "no_speech": ("No speech", "muted"),
}

PLATFORM_LABELS = {
    "google_meet": "Google Meet",
    "zoom": "Zoom",
    "teams": "Teams",
}


# --- template helpers -------------------------------------------------------

def initials(name: str) -> str:
    parts = [p for p in name.split() if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def speaker_hue(name: str) -> int:
    return int(hashlib.sha1(name.encode("utf-8")).hexdigest()[:4], 16) % 360


def fmt_time(dt: datetime | None) -> str:
    return dt.astimezone().strftime("%H:%M") if dt else ""


def iso_utc(dt: datetime | None) -> str:
    """The exact instant, for <time datetime>: fmt_day/fmt_time render the
    server's zone (the stack's TZ), and static/localtime.js rewrites them in
    the viewer's own."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if dt else ""


def fmt_clock(seconds: float | None) -> str:
    if seconds is None:
        return ""
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}" if s < 3600 else f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _user(request: Request) -> auth.SessionUser | None:
    return auth.current_user(settings, request)


def _ui_lang(request: Request | None) -> str:
    """The viewer's interface language: their choice in Settings, English for
    everyone who made none and on every page before sign-in."""
    if request is None:
        return DEFAULT_UI_LANGUAGE
    cached = getattr(request.state, "ui_lang", None)
    if cached:
        return cached
    user = _user(request)
    lang = (get_ui_language(settings.data_dir, user.vexa_user_id) if user else None) or DEFAULT_UI_LANGUAGE
    request.state.ui_lang = lang
    return lang


def _ctx_lang(ctx) -> str:
    return _ui_lang(ctx.get("request"))


templates.env.globals.update(
    initials=initials,
    speaker_hue=speaker_hue,
    fmt_day=pass_context(lambda ctx, dt: format_datetime_day(dt, _ctx_lang(ctx))),
    fmt_time=fmt_time,
    iso_utc=iso_utc,
    fmt_clock=fmt_clock,
    quality_labels=QUALITY_LABELS,
    platform_labels=PLATFORM_LABELS,
    lang_flag=lang_flag,
    lang_label=lang_label,
    lang_name=lang_name,
    t=pass_context(lambda ctx, text, **values: translate(_ctx_lang(ctx), text, **values)),
    th=pass_context(lambda ctx, text, **values: translate_markup(_ctx_lang(ctx), text, **values)),
    tn=pass_context(
        lambda ctx, singular, plural, n, **values: translate_n(_ctx_lang(ctx), singular, plural, n, **values)
    ),
    ui_lang=pass_context(_ctx_lang),
)


def _clipboard_notes(title: str, when: str, insights: dict, lang: str = DEFAULT_UI_LANGUAGE) -> dict:
    """Build the meeting notes as BOTH plain text and HTML for one clipboard
    write. Pasting into Gmail/Outlook picks the HTML (real headings and
    bullets); pasting into a plain editor picks the text. Markdown is
    deliberately avoided: `##` and `*` render as literal noise in an email.

    Sections mirror exactly what the page shows, in the same order, so the
    button never copies something the reader cannot see."""
    import html as _html

    def items(key):
        return [str(x).strip() for x in (insights.get(key) or []) if str(x).strip()]

    sections: list[tuple[str, list[str]]] = []
    summary = (insights.get("summary") or "").strip()
    if summary:
        sections.append(("Summary", [summary]))
    if items("most_important_points"):
        sections.append(("Key points", items("most_important_points")))

    actions = []
    for item in insights.get("action_items") or []:
        task = str(item.get("task") or "").strip()
        if not task:
            continue
        owner = (item.get("owner") or "").strip()
        due = (item.get("due_date") or "").strip()
        suffix = translate(lang, "(due {date})", date=due) if due else ""
        line = f"{owner}: {task}" if owner else task
        actions.append(f"{line} {suffix}".strip())
    if actions:
        sections.append(("Action items", actions))
    if items("decisions"):
        sections.append(("Decisions", items("decisions")))
    if items("risks_or_open_questions"):
        sections.append(("Risks & open questions", items("risks_or_open_questions")))

    header = title if not when else f"{title} — {when}"
    body_text: list[str] = []
    body_html: list[str] = []
    for key, lines in sections:
        name = translate(lang, key)
        body_text.append(f"{name}:")
        body_html.append(f"<p><strong>{_html.escape(name)}</strong></p>")
        if key == "Summary":
            body_text.append(lines[0])
            body_html.append(f"<p>{_html.escape(lines[0])}</p>")
        else:
            body_text.extend(f"  - {line}" for line in lines)
            body_html.append(
                "<ul>" + "".join(f"<li>{_html.escape(line)}</li>" for line in lines) + "</ul>"
            )
        body_text.append("")
    return {
        "text": "\n".join([header, "=" * len(header), "", *body_text]).strip() + "\n",
        "html": f"<p><strong>{_html.escape(header)}</strong></p>" + "".join(body_html),
        # The header carries the server's time; the page rebuilds it with the
        # viewer's (static/localtime.js), so it also needs the parts.
        "title": title,
        "body_text": "\n".join(body_text).strip(),
        "body_html": "".join(body_html),
    }


def _can_view_shared(meeting_id: str, latest: dict | None, user) -> bool:
    """Owner/admin, OR a shared VIEWER (per-meeting grant, possibly from a
    snapshot team share). Viewer access covers reading only — the mutating
    routes (reprocess, rename) keep the strict owner/admin check."""
    if latest is not None and can_view(latest, user.vexa_user_id, user.is_admin):
        return True
    return user.email in sharing.granted_emails(settings.data_dir, meeting_id)


# --- auth routes ------------------------------------------------------------

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if _user(request):
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse(
        request,
        "login.html",
        {
            "google_ready": settings.google_oauth_configured,
            "dev_available": settings.dev_mode_available,
            "dev_user": settings.dev_user,
        },
    )


def _post_login_target(email: str) -> str:
    """First login (no calendar connected yet) lands on onboarding."""
    return "/" if auth.read_calendar_connection(settings, email) else "/welcome"


@app.post("/login/dev")
async def login_dev(request: Request):
    if not settings.dev_mode_available:
        return PlainTextResponse("Dev login unavailable", status_code=403)
    mapped = settings.user_map()[settings.dev_user.strip().lower()]
    response = RedirectResponse(_post_login_target(mapped.email), status_code=303)
    auth.issue_session(settings, response, mapped)
    return response


def _redirect_uri() -> str:
    return settings.base_url.rstrip("/") + "/auth/google/callback"


@app.get("/auth/google")
async def google_login(request: Request):
    if oauth is None:
        return PlainTextResponse("Google OAuth is not configured yet", status_code=503)
    request.session["oauth_flow"] = "login"
    return await oauth.google.authorize_redirect(request, _redirect_uri())


@app.get("/auth/google/callback")
async def google_callback(request: Request):
    """Single registered redirect URI, two flows: login and calendar-connect
    (disambiguated by a session marker set before the outbound redirect)."""
    if oauth is None:
        return PlainTextResponse("Google OAuth is not configured yet", status_code=503)
    flow = request.session.pop("oauth_flow", "login")

    if flow == "calendar":
        user = _user(request)
        if user is None:
            return auth.login_redirect(request)
        token = await oauth.google_cal.authorize_access_token(request)
        try:
            path, account = auth.store_calendar_token(settings, user.email, token)
        except ValueError as exc:
            logger.warning("Calendar connect for %s failed: %s", user.email, exc)
            return RedirectResponse("/settings?error=no_refresh_token", status_code=303)
        logger.info("Calendar connected for %s (account %s) at %s", user.email, account, path)
        # One-time flash consumed by the next meetings-page render — a URL param
        # would resurface the notice on every refresh of that URL.
        request.session["flash_calendar_connected"] = True
        return RedirectResponse("/", status_code=303)

    token = await oauth.google.authorize_access_token(request)
    info = token.get("userinfo") or {}
    return _complete_login(request, str(info.get("email", "")).lower(), info.get("name"))


def _complete_login(request: Request, email: str, name: str | None):
    """Post-OAuth: roster members get a session; anyone else becomes a PENDING
    ACCESS REQUEST for the admin to approve (/admin/users) instead of a flat
    denial — the self-service onboarding flow. A pending request grants
    nothing: no session, no data."""
    if not email:
        return templates.TemplateResponse(request, "denied.html", {"email": email}, status_code=403)
    mapped = settings.user_map().get(email)
    if mapped is None:
        _signup.add(settings.data_dir, email, name)
        logger.info("Access request queued for %s", email)
        return templates.TemplateResponse(
            request, "pending.html", {"email": email}, status_code=200
        )
    response = RedirectResponse(_post_login_target(email), status_code=303)
    auth.issue_session(settings, response, mapped, name=name)
    return response


@app.get("/welcome", response_class=HTMLResponse)
async def welcome(request: Request):
    user = _user(request)
    if user is None:
        return auth.login_redirect(request)
    if auth.read_calendar_connection(settings, user.email):
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse(
        request,
        "welcome.html",
        {
            "user": user,
            "google_ready": settings.google_oauth_configured,
            "summary_languages": SUMMARY_LANGUAGES,
            "summary_language": get_summary_language(settings.data_dir, user.vexa_user_id)
            or "same-as-transcript",
            "autojoin_policies": AUTOJOIN_POLICIES,
            "autojoin_policy": get_autojoin_policy(settings.data_dir, user.email),
        },
    )


# --- settings ----------------------------------------------------------------

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(
    request: Request, connected: str = "", error: str = "",
    voiceprint: str = "", w: str = "",
):
    user = _user(request)
    if user is None:
        return auth.login_redirect(request)
    from . import voiceprint_requests as _vpreq

    my_print = None
    try:
        my_print = _print_for_email(await _worker_prints(), user.email)
    except httpx.HTTPError:
        pass  # worker down: the card degrades to invitation-only info
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "user": user,
            "calendar_account": auth.read_calendar_connection(settings, user.email),
            "google_ready": settings.google_oauth_configured,
            "just_connected": connected == "1",
            "error": error,
            "my_print": my_print,
            "voiceprint_pending": _vpreq.pending_for(settings.data_dir, user.email),
            "voiceprint_enrolled_now": voiceprint == "enrolled",
            "voiceprint_warning": w,
            "notice_version": _vpreq.NOTICE_VERSION,
            "ui_languages": UI_LANGUAGES,
            "ui_language": _ui_lang(request),
            "summary_languages": SUMMARY_LANGUAGES,
            "summary_language": get_summary_language(settings.data_dir, user.vexa_user_id)
            or "same-as-transcript",
            "autojoin_policies": AUTOJOIN_POLICIES,
            "autojoin_policy": get_autojoin_policy(settings.data_dir, user.email),
            "auto_share": get_auto_share(settings.data_dir, user.vexa_user_id),
            "capture_token": _capture_token_for(user),
            # The URL the extension talks to. This very site: /capture/* is
            # proxied to the worker, so one public hostname serves everything
            # (and the worker's port never needs to be reachable from clients).
            "worker_public_url": settings.base_url.rstrip("/"),
            "extension_store_url": settings.extension_store_url,
        },
    )


def _capture_token_for(user) -> str | None:
    from fred_live_worker.capture import mint_capture_token

    if not settings.session_secret:
        return None
    return mint_capture_token(settings.session_secret, user.vexa_user_id, user.email)


@app.post("/settings/autojoin")
async def save_autojoin_policy(request: Request, autojoin_policy: str = Form(...)):
    user = _user(request)
    if user is None:
        return auth.login_redirect(request)
    if autojoin_policy not in AUTOJOIN_POLICIES:
        return PlainTextResponse("Unsupported policy", status_code=400)
    set_autojoin_policy(settings.data_dir, user.vexa_user_id, user.email, autojoin_policy)
    return RedirectResponse("/settings", status_code=303)


@app.post("/settings/ui-language")
async def save_ui_language(request: Request, ui_language: str = Form(...)):
    user = _user(request)
    if user is None:
        return auth.login_redirect(request)
    if ui_language not in UI_LANGUAGES:
        return PlainTextResponse("Unsupported language", status_code=400)
    set_ui_language(settings.data_dir, user.vexa_user_id, ui_language, email=user.email)
    return RedirectResponse("/settings", status_code=303)


@app.post("/settings/auto-share")
async def save_auto_share(request: Request, auto_share: str = Form("")):
    # A checkbox: absent from the form when unticked.
    user = _user(request)
    if user is None:
        return auth.login_redirect(request)
    set_auto_share(settings.data_dir, user.vexa_user_id, auto_share == "on", email=user.email)
    return RedirectResponse("/settings", status_code=303)


@app.post("/settings/language")
async def save_summary_language(request: Request, summary_language: str = Form(...), next_url: str = Form("/settings")):
    user = _user(request)
    if user is None:
        return auth.login_redirect(request)
    if summary_language not in SUMMARY_LANGUAGES:
        return PlainTextResponse("Unsupported language", status_code=400)
    set_summary_language(settings.data_dir, user.vexa_user_id, summary_language, email=user.email)
    if next_url not in ("/settings", "/welcome", "/"):
        next_url = "/settings"
    return RedirectResponse(next_url, status_code=303)


@app.post("/settings/calendar/connect")
async def calendar_connect(
    request: Request,
    summary_language: str = Form(None),
    autojoin_policy: str = Form(None),
):
    user = _user(request)
    if user is None:
        return auth.login_redirect(request)
    if oauth is None:
        return PlainTextResponse("Google OAuth is not configured yet", status_code=503)
    # Onboarding sends the language + auto-join choices along with the connect click.
    if summary_language and summary_language in SUMMARY_LANGUAGES:
        set_summary_language(settings.data_dir, user.vexa_user_id, summary_language, email=user.email)
    if autojoin_policy and autojoin_policy in AUTOJOIN_POLICIES:
        set_autojoin_policy(settings.data_dir, user.vexa_user_id, user.email, autojoin_policy)
    request.session["oauth_flow"] = "calendar"
    # offline + consent forces a refresh token (required: the watcher polls 24/7).
    return await oauth.google_cal.authorize_redirect(
        request, _redirect_uri(), access_type="offline", prompt="consent"
    )


@app.post("/settings/calendar/disconnect")
async def calendar_disconnect(request: Request):
    user = _user(request)
    if user is None:
        return auth.login_redirect(request)
    path = auth.calendar_token_path(settings, user.email)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not remove calendar token %s", path, exc_info=True)
    return RedirectResponse("/settings", status_code=303)


@app.post("/logout")
async def logout():
    response = RedirectResponse("/login", status_code=303)
    auth.clear_session(response)
    return response


# --- pages ------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def meetings_page(request: Request, q: str = "", platform: str = ""):
    user = _user(request)
    if user is None:
        return auth.login_redirect(request)
    all_cards = index.list_cards()
    owned = visible_cards(all_cards, user.vexa_user_id, user.is_admin)
    owned_ids = {c.meeting_id for c in owned}
    # Shared-viewer cards ride along (incl. processing/failed states), badged.
    shared_ids = sharing.meetings_shared_with(settings.data_dir, user.email) - owned_ids
    cards = [c for c in all_cards if c.meeting_id in owned_ids or c.meeting_id in shared_ids]

    # Platform chips reflect the user's whole corpus (pre-search), so a chip
    # never disappears just because the current search excludes its meetings.
    platform_counts: dict[str, int] = {}
    for c in cards:
        if c.platform:
            platform_counts[c.platform] = platform_counts.get(c.platform, 0) + 1
    all_count = len(cards)

    q = q.strip()
    platform = platform.strip()
    if platform:
        cards = [c for c in cards if c.platform == platform]
    if q:
        needle = q.lower()
        cards = [c for c in cards if needle in c.search_blob]

    any_processing = any(c.status == "processing" for c in cards)

    groups: list[dict] = []
    lang = _ui_lang(request)
    for card in cards:
        day = format_datetime_day(card.start_time, lang)
        if not groups or groups[-1]["day"] != day:
            groups.append({"day": day, "cards": []})
        groups[-1]["cards"].append(card)

    return templates.TemplateResponse(
        request,
        "meetings.html",
        {
            "user": user,
            "groups": groups,
            "total": len(cards),
            "q": q,
            "platform": platform,
            "platform_counts": platform_counts,
            "all_count": all_count,
            "any_processing": any_processing,
            "shared_ids": shared_ids,
            "calendar_account": auth.read_calendar_connection(settings, user.email),
            "just_connected": bool(request.session.pop("flash_calendar_connected", False)),
        },
    )


def _speaker_blocks(transcript: dict) -> list[dict]:
    """Group consecutive same-speaker segments for a readable transcript."""
    blocks: list[dict] = []
    for seg in transcript.get("segments") or []:
        speaker = seg.get("speaker") or "Speaker"
        lang = seg.get("language")
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        if blocks and blocks[-1]["speaker"] == speaker:
            blocks[-1]["parts"].append({"text": text, "language": lang})
        else:
            blocks.append(
                {
                    "speaker": speaker,
                    "start": seg.get("start_seconds"),
                    "parts": [{"text": text, "language": lang}],
                }
            )
    return blocks


@app.get("/meetings/{meeting_id}", response_class=HTMLResponse)
async def meeting_page(request: Request, meeting_id: str):
    user = _user(request)
    if user is None:
        return auth.login_redirect(request)
    latest = index.load_latest(meeting_id)
    if latest is None or not _can_view_shared(meeting_id, latest, user):
        return templates.TemplateResponse(
            request, "notfound.html", {"user": user}, status_code=404
        )
    is_owner = can_view(latest, user.vexa_user_id, user.is_admin)

    cards = {c.meeting_id: c for c in index.list_cards()}
    card = cards.get(meeting_id)
    transcript = latest.get("transcript") or {}
    insights = latest.get("insights") or {}
    chat, chat_links = _chat_view(transcript)
    blocks = _speaker_blocks(transcript)
    speakers = list(dict.fromkeys(b["speaker"] for b in blocks))
    lang = _ui_lang(request)
    return templates.TemplateResponse(
        request,
        "meeting.html",
        {
            "user": user,
            "card": card,
            "meeting_id": meeting_id,
            "meeting": latest.get("meeting") or {},
            "quality": latest.get("quality") or "ok",
            "insights": insights,
            "blocks": blocks,
            "speakers": speakers,
            "has_transcript": bool((transcript.get("text") or "").strip()),
            "chat": chat,
            "chat_links": chat_links,
            "recording_notice": _recording_notice(latest.get("meeting") or {}),
            "is_owner": is_owner,
            "clipboard_notes": _clipboard_notes(
                (card.title if card else translate(lang, "Meeting {id}", id=meeting_id)),
                (f"{format_datetime_day(card.start_time, lang)} {fmt_time(card.start_time)}".strip()
                 if card and card.start_time else ""),
                insights,
                lang,
            ),
            "grants": sharing.grants_for(settings.data_dir, meeting_id),
            "teams": sharing.load_teams(settings.data_dir) if is_owner else {},
            "share_candidates": sorted(settings.user_map()) if is_owner else [],
        },
    )


def _chat_view(transcript: dict) -> tuple[list[dict], list[dict]]:
    """Chat messages and the distinct links posted in them, re-validated here:
    links become clickable anchors for people who did not write them, so the
    page never trusts what reached latest.json (a run from before validation,
    or a hand-edited file)."""
    from fred_live_worker.meeting_chat import safe_link

    messages = []
    links: list[dict] = []
    seen: set[str] = set()
    for raw in transcript.get("chat_messages") or []:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text") or "").strip()
        if not text:
            continue
        offset = raw.get("offset_seconds")
        offset = float(offset) if isinstance(offset, (int, float)) else None
        sender = str(raw.get("sender") or "Unknown")
        safe = [u for u in (safe_link(l) for l in (raw.get("links") or [])) if u]
        messages.append({"sender": sender, "text": text, "offset": offset, "links": safe})
        for url in safe:
            if url not in seen:
                seen.add(url)
                links.append({"url": url, "sender": sender, "offset": offset})
    return messages, links


def _recording_notice(meeting: dict) -> dict | None:
    """A warning for the meeting page when the chat notice (v0.3.12+) did NOT
    reach the participants: it failed, or the page had no chat to post to.
    A notice that went out, or one the recorder turned off, stays on record in
    result.json but is not shown (the page is about
    the meeting). None too for bot meetings and older extensions, which have
    nothing on record."""
    notice = ((meeting or {}).get("data") or {}).get("recording_notice")
    if not isinstance(notice, dict):
        return None
    events = [e for e in (notice.get("events") or []) if isinstance(e, dict)]
    if any(e.get("status") == "sent" for e in events):
        return None
    if notice.get("enabled") is False or any(e.get("status") == "disabled" for e in events):
        return None
    failed = [e for e in events if e.get("status") == "failed"]
    if failed:
        return {"state": "failed", "reason": failed[-1].get("reason")}
    if any(e.get("status") == "unsupported" for e in events):
        return {"state": "unsupported"}
    return None


@app.get("/meetings/{meeting_id}/transcript.txt")
async def transcript_download(request: Request, meeting_id: str):
    user = _user(request)
    if user is None:
        return auth.login_redirect(request)
    latest = index.load_latest(meeting_id)
    if latest is None or not _can_view_shared(meeting_id, latest, user):
        return PlainTextResponse("Not found", status_code=404)
    lines = []
    for block in _speaker_blocks(latest.get("transcript") or {}):
        text = " ".join(p["text"] for p in block["parts"])
        lines.append(f"{block['speaker']}: {text}")
    chat, _links = _chat_view(latest.get("transcript") or {})
    if chat:
        from fred_live_worker.meeting_chat import fmt_offset

        lines.append("--- Meeting chat (typed messages) ---")
        lines.append("\n".join(f"[{fmt_offset(m['offset'])}] {m['sender']}: {m['text']}" for m in chat))
    return PlainTextResponse(
        "\n\n".join(lines),
        headers={"Content-Disposition": f'attachment; filename="meeting-{meeting_id}.txt"'},
    )


def _card_owned(meeting_id: str, user) -> bool:
    card = next((c for c in index.list_cards() if c.meeting_id == meeting_id), None)
    if card is None:
        return False
    return user.is_admin or (
        card.vexa_user_id is not None and card.vexa_user_id == user.vexa_user_id
    )


@app.post("/meetings/{meeting_id}/share")
async def meeting_share(
    request: Request, meeting_id: str, email: str = Form(""), team: str = Form("")
):
    """Owner/admin only. Direct grants must be registered users; team shares
    expand to the CURRENT member list (snapshot semantics)."""
    user = _user(request)
    if user is None:
        return auth.login_redirect(request)
    if not _card_owned(meeting_id, user):
        return PlainTextResponse(translate(_ui_lang(request), "Only the owner can share"), status_code=403)
    email = email.strip().lower()
    if email:
        if email not in settings.user_map():
            return RedirectResponse(f"/meetings/{meeting_id}?share_error=not-registered", status_code=303)
        sharing.grant(settings.data_dir, meeting_id, [email], granted_by=user.email)
        logger.info("Meeting %s shared with %s by %s", meeting_id, email, user.email)
    elif team:
        added = sharing.grant_team(settings.data_dir, meeting_id, team.strip(), granted_by=user.email)
        logger.info("Meeting %s shared with team %s (%d grants) by %s", meeting_id, team, added, user.email)
    return RedirectResponse(f"/meetings/{meeting_id}", status_code=303)


@app.post("/meetings/{meeting_id}/unshare")
async def meeting_unshare(request: Request, meeting_id: str, email: str = Form(...)):
    user = _user(request)
    if user is None:
        return auth.login_redirect(request)
    if not _card_owned(meeting_id, user):
        return PlainTextResponse(translate(_ui_lang(request), "Only the owner can share"), status_code=403)
    sharing.revoke(settings.data_dir, meeting_id, email)
    logger.info("Meeting %s: access revoked for %s by %s", meeting_id, email, user.email)
    return RedirectResponse(f"/meetings/{meeting_id}", status_code=303)


# --- ask chat (HTMX partial) -------------------------------------------------

MAX_QUESTION_CHARS = 2000
MAX_HISTORY_MESSAGES = 10


@app.post("/meetings/{meeting_id}/ask", response_class=HTMLResponse)
async def ask(request: Request, meeting_id: str, question: str = Form(...), history: str = Form("[]")):
    user = _user(request)
    if user is None:
        return PlainTextResponse(translate(_ui_lang(request), "Session expired — reload the page"), status_code=401)
    latest = index.load_latest(meeting_id)
    if latest is None or not _can_view_shared(meeting_id, latest, user):
        return PlainTextResponse("Not found", status_code=404)

    question = question.strip()[:MAX_QUESTION_CHARS]
    if not question:
        return HTMLResponse("")
    try:
        past = json.loads(history)
        assert isinstance(past, list)
    except (json.JSONDecodeError, AssertionError):
        past = []
    past = [
        m for m in past
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
    ][-MAX_HISTORY_MESSAGES:]

    headers = {}
    if settings.worker_token:
        headers["X-Fred-Live-Token"] = settings.worker_token
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(
                f"{settings.worker_url.rstrip('/')}/internal/meetings/{meeting_id}/ask",
                json={"question": question, "history": past},
                headers=headers,
            )
        resp.raise_for_status()
        answer = resp.json().get("answer", "")
        error = None
    except httpx.HTTPError as exc:
        logger.warning("Ask proxy failed for meeting %s: %s", meeting_id, exc)
        answer, error = "", translate(_ui_lang(request), "The assistant is unavailable right now — is the worker/LLM running?")

    new_history = past + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer or (error or "")},
    ]
    return templates.TemplateResponse(
        request,
        "partials/ask_exchange.html",
        {
            "question": question,
            "answer": answer,
            "error": error,
            "history_json": json.dumps(new_history[-MAX_HISTORY_MESSAGES:], ensure_ascii=False),
        },
    )


@app.post("/meetings/{meeting_id}/reprocess")
async def reprocess(request: Request, meeting_id: str):
    """Proxy the web Reprocess button to the worker (token stays server-side)."""
    user = _user(request)
    if user is None:
        return PlainTextResponse(translate(_ui_lang(request), "Session expired — reload the page"), status_code=401)
    latest = index.load_latest(meeting_id)
    if latest is not None:
        allowed = can_view(latest, user.vexa_user_id, user.is_admin)
    else:
        # Failed-before-first-result meetings have no latest.json; scope via the
        # card built from the saved hook.
        card = next((c for c in index.list_cards() if c.meeting_id == meeting_id), None)
        allowed = card is not None and (user.is_admin or card.vexa_user_id == user.vexa_user_id)
    if not allowed:
        return PlainTextResponse("Not found", status_code=404)

    headers = {}
    if settings.worker_token:
        headers["X-Fred-Live-Token"] = settings.worker_token
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{settings.worker_url.rstrip('/')}/internal/meetings/{meeting_id}/reprocess",
                headers=headers,
            )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("Reprocess proxy failed for meeting %s: %s", meeting_id, exc)
        return PlainTextResponse("Worker unreachable — is it running?", status_code=502)
    return RedirectResponse("/", status_code=303)


@app.post("/meetings/{meeting_id}/speakers")
async def rename_speakers(request: Request, meeting_id: str):
    """Batch speaker rename: parallel original[]/updated[] form fields; only
    changed pairs are sent to the worker (token stays server-side)."""
    user = _user(request)
    if user is None:
        return auth.login_redirect(request)
    latest = index.load_latest(meeting_id)
    if latest is None or not can_view(latest, user.vexa_user_id, user.is_admin):
        return PlainTextResponse("Not found", status_code=404)

    form = await request.form()
    originals = form.getlist("original")
    updated = form.getlist("updated")
    renames = {
        o.strip(): n.strip()
        for o, n in zip(originals, updated)
        if o.strip() and n.strip() and o.strip() != n.strip()
    }
    if not renames:
        return RedirectResponse(f"/meetings/{meeting_id}", status_code=303)

    headers = {}
    if settings.worker_token:
        headers["X-Fred-Live-Token"] = settings.worker_token
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{settings.worker_url.rstrip('/')}/internal/meetings/{meeting_id}/rename-speakers",
                json={"renames": renames},
                headers=headers,
            )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("Speaker rename failed for meeting %s: %s", meeting_id, exc)
        return PlainTextResponse("Rename failed — is the worker running?", status_code=502)
    return RedirectResponse(f"/meetings/{meeting_id}", status_code=303)




# --- voiceprints: leader invitations + member self-enrollment ----------------
# GDPR-shaped flow: the admin (team leader) only INVITES and can delete; each
# member records/uploads their OWN audio behind an explicit consent notice.
# Embeddings are computed by the worker; raw audio is deleted right after.

from . import voiceprint_requests as vpreq


def _worker_headers() -> dict:
    headers = {}
    if settings.worker_token:
        headers["X-Fred-Live-Token"] = settings.worker_token
    return headers


async def _worker_prints() -> list[dict]:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{settings.worker_url.rstrip('/')}/internal/voiceprints",
            headers=_worker_headers(),
        )
    resp.raise_for_status()
    return resp.json().get("prints", [])


def _print_for_email(prints: list[dict], email: str) -> dict | None:
    email = email.strip().lower()
    for p in prints:
        if (p.get("metadata") or {}).get("email", "").lower() == email:
            return p
    return None


@app.get("/admin/voiceprints", response_class=HTMLResponse)
async def admin_voiceprints(request: Request, error: str = ""):
    user = _user(request)
    if user is None:
        return auth.login_redirect(request)
    if not user.is_admin:
        return PlainTextResponse("Admins only", status_code=403)
    try:
        prints = await _worker_prints()
    except httpx.HTTPError:
        prints = []
        error = error or "Worker unreachable; showing invitations only."
    requests_state = vpreq.all_requests(settings.data_dir)
    roster = []
    linked_slugs = set()
    for email, entry in sorted(settings.user_map().items()):
        print_ = _print_for_email(prints, email)
        if print_:
            linked_slugs.add(print_.get("slug"))
        roster.append({
            "email": email,
            "is_admin": entry.is_admin,
            "print": print_,
            "request": requests_state.get(email),
        })
    unlinked = [p for p in prints if p.get("slug") not in linked_slugs]
    # One print per account: only members without one can be linked.
    linkable = [row["email"] for row in roster if not row["print"]]
    return templates.TemplateResponse(
        request,
        "admin_voiceprints.html",
        {"user": user, "roster": roster, "unlinked": unlinked, "linkable": linkable, "error": error},
    )


@app.post("/admin/voiceprints/request")
async def admin_voiceprint_request(request: Request, email: str = Form(...)):
    user = _user(request)
    if user is None:
        return auth.login_redirect(request)
    if not user.is_admin:
        return PlainTextResponse("Admins only", status_code=403)
    email = email.strip().lower()
    if email not in settings.user_map():
        return PlainTextResponse("Unknown member", status_code=400)
    vpreq.request_enrollment(settings.data_dir, email, user.email)
    return RedirectResponse("/admin/voiceprints", status_code=303)


@app.post("/admin/voiceprints/cancel")
async def admin_voiceprint_cancel(request: Request, email: str = Form(...)):
    user = _user(request)
    if user is None:
        return auth.login_redirect(request)
    if not user.is_admin:
        return PlainTextResponse("Admins only", status_code=403)
    vpreq.clear(settings.data_dir, email)
    return RedirectResponse("/admin/voiceprints", status_code=303)


@app.post("/admin/voiceprints/link")
async def admin_voiceprint_link(request: Request, slug: str = Form(...), email: str = Form("")):
    """Attach a print to a Fred account (prints from the operator CLI carry
    no email), so the person can be shared meetings automatically."""
    user = _user(request)
    if user is None:
        return auth.login_redirect(request)
    if not user.is_admin:
        return PlainTextResponse("Admins only", status_code=403)
    email = email.strip().lower()
    if email and email not in settings.user_map():
        return PlainTextResponse("Unknown member", status_code=400)
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{settings.worker_url.rstrip('/')}/internal/voiceprints/{slug}/account",
                json={"email": email or None, "by": user.email},
                headers=_worker_headers(),
            )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("Voiceprint link failed: %s", exc)
        return RedirectResponse("/admin/voiceprints?error=Link+failed", status_code=303)
    return RedirectResponse("/admin/voiceprints", status_code=303)


@app.post("/admin/voiceprints/delete")
async def admin_voiceprint_delete(request: Request, slug: str = Form(...)):
    user = _user(request)
    if user is None:
        return auth.login_redirect(request)
    if not user.is_admin:
        return PlainTextResponse("Admins only", status_code=403)
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.delete(
                f"{settings.worker_url.rstrip('/')}/internal/voiceprints/{slug}",
                headers=_worker_headers(),
            )
        if resp.status_code not in (200, 404):
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("Voiceprint delete failed: %s", exc)
        return RedirectResponse("/admin/voiceprints?error=Delete+failed", status_code=303)
    return RedirectResponse("/admin/voiceprints", status_code=303)


@app.post("/settings/voiceprint")
async def settings_voiceprint_enroll(
    request: Request,
    display_name: str = Form(...),
    source: str = Form("self-recorded"),
    audio: UploadFile = File(...),
):
    """Member enrolls THEIR OWN voice (consent checkbox enforced client-side,
    recorded server-side via the consent metadata on the print)."""
    user = _user(request)
    if user is None:
        return auth.login_redirect(request)
    display_name = display_name.strip()[:80]
    if not display_name:
        return PlainTextResponse(translate(_ui_lang(request), "Name is required"), status_code=400)
    payload = await audio.read()
    if len(payload) > 30 * 1024 * 1024:
        return PlainTextResponse(translate(_ui_lang(request), "Audio too large"), status_code=413)
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{settings.worker_url.rstrip('/')}/internal/voiceprints/enroll",
                headers=_worker_headers(),
                data={
                    "name": display_name,
                    "email": user.email,
                    "source": source if source in ("self-recorded", "upload") else "upload",
                    "notice_version": vpreq.NOTICE_VERSION,
                },
                files={"audio": (audio.filename or "sample.webm", payload)},
            )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("Voiceprint enrollment failed for %s: %s", user.email, exc)
        return PlainTextResponse(
            translate(_ui_lang(request), "Enrollment failed (is the worker running? was the audio valid?)"),
            status_code=502,
        )
    vpreq.clear(settings.data_dir, user.email)
    warning = (resp.json() or {}).get("warning")
    suffix = "&warning=" + httpx.QueryParams({"w": warning})["w"] if warning else ""
    return RedirectResponse(f"/settings?voiceprint=enrolled{suffix}", status_code=303)


@app.post("/settings/voiceprint/decline")
async def settings_voiceprint_decline(request: Request):
    user = _user(request)
    if user is None:
        return auth.login_redirect(request)
    vpreq.decline(settings.data_dir, user.email)
    return RedirectResponse("/settings", status_code=303)


@app.post("/settings/voiceprint/delete")
async def settings_voiceprint_delete(request: Request):
    """Withdrawal: a member deletes their OWN print (matched by email)."""
    user = _user(request)
    if user is None:
        return auth.login_redirect(request)
    try:
        prints = await _worker_prints()
        mine = _print_for_email(prints, user.email)
        if mine:
            async with httpx.AsyncClient(timeout=15) as client:
                await client.delete(
                    f"{settings.worker_url.rstrip('/')}/internal/voiceprints/{mine['slug']}",
                    headers=_worker_headers(),
                )
    except httpx.HTTPError as exc:
        logger.warning("Self voiceprint delete failed for %s: %s", user.email, exc)
        return PlainTextResponse(translate(_ui_lang(request), "Delete failed; try again"), status_code=502)
    return RedirectResponse("/settings", status_code=303)



@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


# --- admin: users (self-service signup approvals + membership) ---------------

def _admin_or_none(request: Request):
    user = _user(request)
    if user is None or not user.is_admin:
        return None
    return user


@app.get("/admin/users", response_class=HTMLResponse)
async def admin_users(request: Request, error: str = ""):
    user = _admin_or_none(request)
    if user is None:
        return auth.login_redirect(request) if _user(request) is None else PlainTextResponse(
            "Admins only", status_code=403
        )
    from fred_live_worker.users import parse_env_users

    env_emails = set(parse_env_users(settings.users))
    members = [
        {"email": email, "vexa_user_id": u.vexa_user_id, "is_admin": u.is_admin,
         "from_env": email in env_emails, "is_self": email == user.email}
        for email, u in sorted(settings.user_map().items())
    ]
    pending = [
        {"email": email, **entry}
        for email, entry in sorted(_signup.all_requests(settings.data_dir).items())
        if email not in {m["email"] for m in members}
    ]
    prefix = "Approval failed: "  # the one message with a detail in it (admin_users_approve)
    if error.startswith(prefix):
        error = translate(_ui_lang(request), "Approval failed: {detail}", detail=error[len(prefix):])
    return templates.TemplateResponse(
        request, "admin_users.html",
        {"user": user, "members": members, "pending": pending, "error": error,
         "teams": sharing.load_teams(settings.data_dir),
         "roster_emails": sorted(settings.user_map())},
    )


@app.post("/admin/users/approve")
async def admin_users_approve(request: Request, email: str = Form(...)):
    user = _admin_or_none(request)
    if user is None:
        return PlainTextResponse("Admins only", status_code=403)
    email = email.strip().lower()
    entry = _signup.all_requests(settings.data_dir).get(email) or {}
    try:
        async with httpx.AsyncClient(timeout=90) as client:
            resp = await client.post(
                f"{settings.worker_url.rstrip('/')}/internal/admin/onboard",
                headers=_worker_headers(),
                json={"email": email, "name": entry.get("name"), "admin": False},
            )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        detail = getattr(getattr(exc, "response", None), "text", "") or str(exc)
        logger.warning("Approve failed for %s: %s", email, detail[:300])
        return RedirectResponse(
            f"/admin/users?error=Approval+failed:+{httpx.QueryParams({'e': detail[:120]})['e']}",
            status_code=303,
        )
    _signup.clear(settings.data_dir, email)
    logger.info("Admin %s approved %s", user.email, email)
    return RedirectResponse("/admin/users", status_code=303)


@app.post("/admin/users/reject")
async def admin_users_reject(request: Request, email: str = Form(...)):
    user = _admin_or_none(request)
    if user is None:
        return PlainTextResponse("Admins only", status_code=403)
    _signup.clear(settings.data_dir, email)
    logger.info("Admin %s rejected access request from %s", user.email, email)
    return RedirectResponse("/admin/users", status_code=303)


@app.post("/admin/users/remove")
async def admin_users_remove(request: Request, email: str = Form(...)):
    user = _admin_or_none(request)
    if user is None:
        return PlainTextResponse("Admins only", status_code=403)
    from fred_live_worker.users import parse_env_users

    email = email.strip().lower()
    if email == user.email:
        return RedirectResponse("/admin/users?error=You+cannot+remove+yourself", status_code=303)
    if email in parse_env_users(settings.users):
        return RedirectResponse(
            "/admin/users?error=Env-configured+users+are+removed+in+.env.worker", status_code=303
        )
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{settings.worker_url.rstrip('/')}/internal/admin/offboard",
                headers=_worker_headers(),
                json={"email": email},
            )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("Offboard failed for %s: %s", email, exc)
        return RedirectResponse("/admin/users?error=Remove+failed;+try+again", status_code=303)
    logger.info("Admin %s removed %s", user.email, email)
    return RedirectResponse("/admin/users", status_code=303)


@app.post("/admin/teams/save")
async def admin_teams_save(
    request: Request,
    name: str = Form(...),
    members: list[str] = Form(default=[]),
    original_slug: str = Form(""),
):
    user = _admin_or_none(request)
    if user is None:
        return PlainTextResponse("Admins only", status_code=403)
    # Only registered users can be team members (the chip picker suggests only
    # those; this enforces it against hand-crafted POSTs too).
    roster = settings.user_map()
    members = [m for m in members if m.strip().lower() in roster]
    slug = sharing.save_team(settings.data_dir, name, members)
    # Editing may rename: the form carries the team's previous slug; a changed
    # name creates a new slug, so drop the old entry to complete the rename.
    if slug and original_slug and original_slug != slug:
        sharing.delete_team(settings.data_dir, original_slug)
    logger.info("Team %r saved by %s (%d member(s))", slug, user.email, len(members))
    return RedirectResponse("/admin/users", status_code=303)


@app.post("/admin/teams/delete")
async def admin_teams_delete(request: Request, slug: str = Form(...)):
    user = _admin_or_none(request)
    if user is None:
        return PlainTextResponse("Admins only", status_code=403)
    sharing.delete_team(settings.data_dir, slug)
    logger.info("Team %r deleted by %s", slug, user.email)
    return RedirectResponse("/admin/users", status_code=303)


# --- extension capture proxy ------------------------------------------------
# The Chrome extension records in the coworker's browser and uploads here; we
# forward verbatim to the worker's /capture/* endpoints. This makes the web
# hostname the ONLY public surface: the worker's port stays unreachable from
# clients, and Traefik needs no second route. No session auth on purpose — the
# worker itself validates the per-user capture token on every request (and
# re-checks the allowlist), and capture clients hold that token, not a cookie.

_CAPTURE_FORWARD_HEADERS = ("authorization", "content-type")


async def _forward_capture(method: str, url: str, headers: dict, body: bytes):
    """One place to stub in tests. finish concatenates parts with ffmpeg, so
    the read timeout is generous."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(10, read=300)) as client:
        resp = await client.request(method, url, headers=headers, content=body)
        return resp.status_code, resp.content, resp.headers.get("content-type", "application/json")


@app.post("/capture/{path:path}")
async def capture_proxy(path: str, request: Request):
    headers = {
        k: v for k, v in request.headers.items() if k.lower() in _CAPTURE_FORWARD_HEADERS
    }
    # Keep the query string: chunk uploads carry ?part=<nonce>&seq=<n>, which
    # is what lets the worker drop duplicate retries and a previous recorder's
    # late chunks (append_chunk_dedup). Dropping it here silently put every
    # capture on the legacy append-everything path (field, 2026-10-23: no
    # session had ever recorded a last_seq).
    query = request.url.query
    target = f"{settings.worker_url.rstrip('/')}/capture/{path}" + (f"?{query}" if query else "")
    try:
        status, content, content_type = await _forward_capture(
            "POST",
            target,
            headers,
            await request.body(),
        )
    except httpx.HTTPError as exc:
        logger.warning("Capture proxy failed for /capture/%s: %s", path, exc)
        return PlainTextResponse("Worker unreachable", status_code=502)
    return Response(content=content, status_code=status, media_type=content_type)
