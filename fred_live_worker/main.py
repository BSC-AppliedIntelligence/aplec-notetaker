from __future__ import annotations

import json
import logging
import re
import secrets
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field

from .config import get_settings
from .pipeline import MeetingProcessor
from .schemas import MeetingCompletedPayload, TranscriptResult
from .meeting_chat import transcript_for_llm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

settings = get_settings()
processor = MeetingProcessor(settings)
# Interactive docs (/docs, /redoc, /openapi.json) are disabled: the worker
# binds 0.0.0.0 and this is an internal service, not a public API surface.
app = FastAPI(title="fred-live worker", version="0.1.0",
              docs_url=None, redoc_url=None, openapi_url=None)
logger = logging.getLogger(__name__)


@app.on_event("startup")
async def _startup_recovery() -> None:
    import asyncio

    from .orphan_recordings import schedule_orphan_sweep
    from .recovery import schedule_recovery
    from .retention import schedule_retention

    schedule_recovery(processor, settings)
    schedule_retention(settings)
    schedule_orphan_sweep(processor, settings)

    async def _capture_sweep_loop() -> None:
        await asyncio.sleep(180)
        while True:
            try:
                await sweep_stale_captures()
            except Exception:
                logger.exception("Capture sweep failed")
            await asyncio.sleep(600)

    asyncio.get_event_loop().create_task(_capture_sweep_loop())


async def sweep_stale_captures() -> int:
    """Auto-finish captures the extension abandoned (client crash, killed tab,
    stuck REC): with audio -> process like a normal finish; empty -> just mark
    finished so it stops being scanned. Returns number of sessions handled."""
    from .capture import CaptureStore

    store = CaptureStore(settings)
    handled = 0
    for meta in store.stale_unfinished_sessions(older_than_minutes=15):
        session_id = meta["session_id"]
        try:
            if store.has_audio(session_id):
                logger.warning("Capture %s was never finished; auto-finishing", session_id)
                payload = MeetingCompletedPayload.model_validate(store.finish_payload(session_id))
                await _process_in_background(payload)
            else:
                store.mark_finished(session_id)
            handled += 1
        except Exception:
            logger.exception("Failed to sweep capture %s", session_id)
    return handled


def _require_token(header_token: str | None, query_token: str | None) -> None:
    # Fail CLOSED: the worker binds 0.0.0.0, so /internal/* is the LAN's only
    # gate. An unset or empty token is a misconfiguration, not "auth off":
    # refuse to serve rather than expose every meeting to the network. (An
    # empty configured secret used to accept `?token=` because
    # compare_digest("","") is true; refusing empty expected closes that too.)
    expected = settings.auth_token.get_secret_value() if settings.auth_token else ""
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Worker auth token not configured (set FRED_LIVE_AUTH_TOKEN)",
        )
    provided = header_token or query_token
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing token")


async def _process_in_background(payload: MeetingCompletedPayload) -> None:
    try:
        await processor.process(payload)
    except Exception:
        logger.exception("Failed to process post-meeting payload for meeting %s", payload.meeting.id)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/internal/meetings/{meeting_id}/latest")
async def latest_meeting_result(
    meeting_id: str,
    x_fred_live_token: str | None = Header(default=None),
    token: str | None = Query(default=None),
):
    _require_token(x_fred_live_token, token)
    if not re.fullmatch(r"[\w-]+", meeting_id):
        raise HTTPException(status_code=404, detail=f"No result found for meeting {meeting_id}")
    latest_path = settings.meetings_dir / meeting_id / "latest.json"
    if not latest_path.exists():
        raise HTTPException(status_code=404, detail=f"No result found for meeting {meeting_id}")
    return JSONResponse(content=json.loads(latest_path.read_text(encoding="utf-8")))


@app.post("/internal/meetings/{meeting_id}/reprocess", status_code=202)
async def reprocess_meeting(
    meeting_id: str,
    background_tasks: BackgroundTasks,
    x_fred_live_token: str | None = Header(default=None),
    token: str | None = Query(default=None),
):
    """Re-run the full pipeline from the newest saved hook.json (used by the web
    UI's Reprocess button after a failed run). Bypasses the completion dedup —
    this is an explicit operator request."""
    _require_token(x_fred_live_token, token)
    if not re.fullmatch(r"[\w-]+", meeting_id):
        raise HTTPException(status_code=404, detail=f"No runs found for meeting {meeting_id}")
    meeting_dir = settings.meetings_dir / meeting_id
    if not meeting_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"No runs found for meeting {meeting_id}")
    for run_dir in sorted((d for d in meeting_dir.iterdir() if d.is_dir()), reverse=True):
        hook_path = run_dir / "hook.json"
        if not hook_path.is_file():
            continue
        try:
            payload = MeetingCompletedPayload.model_validate(
                json.loads(hook_path.read_text(encoding="utf-8"))
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"Saved hook is unreadable: {exc}") from exc
        background_tasks.add_task(_process_in_background, payload)
        return {"status": "accepted", "meeting_id": meeting_id, "source_run": run_dir.name}
    raise HTTPException(status_code=404, detail=f"No saved hook for meeting {meeting_id}")


# --- browser-extension capture ingest ----------------------------------------

def _capture_user(authorization: str | None) -> dict:
    from .capture import allowed_capture_user, verify_capture_token

    if not settings.web_session_secret:
        raise HTTPException(status_code=503, detail="Capture tokens not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing capture token")
    user = verify_capture_token(settings.web_session_secret, authorization[len("Bearer "):].strip())
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid capture token")
    # Re-check against the current allowlist (env merged with data/users.json)
    # so removing an email revokes the long-lived capture token, not just the
    # web session. Only enforced when any allowlist is configured (the live
    # deployment sets FRED_LIVE_WEB_USERS).
    from .users import merged_users

    if merged_users(settings.web_users, settings.data_dir) and not allowed_capture_user(
        settings.web_users, user, settings.data_dir
    ):
        raise HTTPException(status_code=403, detail="Capture token no longer authorized")
    return user


def _capture_store():
    from .capture import CaptureStore

    return CaptureStore(settings)


class CaptureStartRequest(BaseModel):
    platform: str = "google_meet"
    native_meeting_id: str | None = None
    title: str | None = None
    # v0.3.8+: identifies THIS recorder instance's chunk stream (see
    # CaptureStore.append_chunk_dedup). Absent from older clients.
    part_nonce: int | None = None
    # v0.3.11+: the extension's own manifest version. Nothing in the system
    # knew which build a capture came from, so "has everyone updated?" could
    # only be answered by waiting out the store's propagation window.
    ext_version: str | None = None
    # v0.3.12+: whether the recorder has the automatic chat notice on, and its
    # text. Absent from older clients, which the meeting page shows as
    # "no notice on record" rather than "turned off".
    notice_enabled: bool | None = None
    notice_text: str | None = Field(default=None, max_length=2000)
    # v0.3.14+: what the recording's channels hold, left then right. Absent from
    # older clients, whose recording is one mono mix.
    audio_channels: list[str] | None = Field(default=None, max_length=2)


@app.post("/capture/start", status_code=201)
async def capture_start(
    body: CaptureStartRequest,
    authorization: str | None = Header(default=None),
):
    from .capture import record_captured_code

    user = _capture_user(authorization)
    store = _capture_store()
    session_id, resumed = store.start(
        user, body.platform, body.native_meeting_id or "", body.title,
        part_nonce=body.part_nonce, ext_version=body.ext_version,
        notice_enabled=body.notice_enabled, notice_text=body.notice_text,
        audio_channels=body.audio_channels,
    )
    record_captured_code(settings.data_dir, body.native_meeting_id)
    if body.title and body.native_meeting_id:
        from datetime import datetime, timezone

        from .autojoin import record_meet_title

        record_meet_title(settings.data_dir, body.native_meeting_id, body.title,
                          datetime.now(timezone.utc))
    return {"session_id": session_id, "resumed": resumed}


@app.post("/capture/{session_id}/chunk")
async def capture_chunk(
    session_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
    part: int | None = Query(default=None),
    seq: int | None = Query(default=None),
):
    from .capture import MAX_CHUNK_BYTES

    user = _capture_user(authorization)
    store = _capture_store()
    try:
        store.check_owner(session_id, user)
    except (ValueError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="Unknown capture session")
    except PermissionError:
        raise HTTPException(status_code=403, detail="Not your capture session")
    data = await request.body()
    if not data:
        return {"received": 0}
    if len(data) > MAX_CHUNK_BYTES:
        raise HTTPException(status_code=413, detail="Chunk too large")
    cap_mb = settings.capture_max_session_mb
    if cap_mb > 0 and store.session_bytes(session_id) + len(data) > cap_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Capture size limit reached")
    if part is not None and seq is not None:
        # Retrying (v0.3.8+) client: at-least-once uploads deduped by
        # (part nonce, seq); genuinely-new audio reopens an auto-finished
        # session so the eventual real finish processes everything.
        status_, total = store.append_chunk_dedup(session_id, data, part, seq)
        if status_ != "appended":
            logger.info("Capture %s chunk dropped (%s): part=%s seq=%s",
                        session_id, status_, part, seq)
            return {"received": 0, "status": status_}
        return {"received": len(data), "total_bytes": total}
    total = store.append_chunk(session_id, data)
    return {"received": len(data), "total_bytes": total}


class CaptureEventsRequest(BaseModel):
    events: list[dict] = Field(default_factory=list, max_length=500)
    # page self-report (participants visible, events counted); logged so
    # no-speaker-event captures are debuggable without the user's console
    diagnostics: dict | None = None


@app.post("/capture/{session_id}/events")
async def capture_events(
    session_id: str,
    body: CaptureEventsRequest,
    authorization: str | None = Header(default=None),
):
    user = _capture_user(authorization)
    store = _capture_store()
    try:
        meta = store.check_owner(session_id, user)
    except (ValueError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="Unknown capture session")
    except PermissionError:
        raise HTTPException(status_code=403, detail="Not your capture session")
    accepted = store.append_events(session_id, body.events, meta)
    if body.diagnostics:
        store.append_diagnostics(session_id, body.diagnostics)
    return {"accepted": accepted}


def _owned_session(session_id: str, authorization: str | None) -> tuple[CaptureStore, dict]:
    user = _capture_user(authorization)
    store = _capture_store()
    try:
        meta = store.check_owner(session_id, user)
    except (ValueError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="Unknown capture session")
    except PermissionError:
        raise HTTPException(status_code=403, detail="Not your capture session")
    return store, meta


class CaptureChatRequest(BaseModel):
    messages: list[dict] = Field(default_factory=list, max_length=500)


@app.post("/capture/{session_id}/chat")
async def capture_chat(
    session_id: str,
    body: CaptureChatRequest,
    authorization: str | None = Header(default=None),
):
    """Messages typed in the meeting's chat panel (v0.3.12+). Kept apart from
    speaker events: a chat sender is not a speaking identity, and counting
    one would raise the diarization floor (see drop_nonspeaker_events)."""
    store, meta = _owned_session(session_id, authorization)
    return {"accepted": store.append_chat(session_id, body.messages, meta)}


class CaptureNoticeRequest(BaseModel):
    status: str = Field(max_length=20)
    kind: str = Field(default="start", max_length=20)
    reason: str | None = Field(default=None, max_length=500)
    relative_ts_ms: int | None = None
    absolute_ts_ms: int | None = None


@app.post("/capture/{session_id}/notice")
async def capture_notice(
    session_id: str,
    body: CaptureNoticeRequest,
    authorization: str | None = Header(default=None),
):
    """One outcome of the recording notice the extension posts in the meeting
    chat — the audit trail the meeting page shows."""
    store, meta = _owned_session(session_id, authorization)
    try:
        entry = store.record_notice(session_id, body.model_dump(), meta)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"recorded": entry}


@app.post("/capture/{session_id}/finish", status_code=202)
async def capture_finish(
    session_id: str,
    background_tasks: BackgroundTasks,
    authorization: str | None = Header(default=None),
):
    user = _capture_user(authorization)
    store = _capture_store()
    try:
        meta = store.check_owner(session_id, user)
        # Idempotent: a finish for an ALREADY-finished session with no new
        # audio must not reprocess (field: the extension re-sent finish ~40min
        # late when a stale tab closed, burning a full GPU pass on identical
        # audio). A resume clears finished_at_ms when it adds a part, so
        # legitimate finish-after-resume still processes.
        if meta.get("finished_at_ms"):
            logger.info("Capture %s finish ignored: already finished", session_id)
            return {"status": "already-finished", "meeting_id": session_id}
        payload_dict = store.finish_payload(session_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc) or "Unknown capture session")
    except PermissionError:
        raise HTTPException(status_code=403, detail="Not your capture session")
    payload = MeetingCompletedPayload.model_validate(payload_dict)
    background_tasks.add_task(_process_in_background, payload)
    logger.info("Capture %s finished; processing enqueued", session_id)
    return {"status": "accepted", "meeting_id": session_id}


class OnboardRequest(BaseModel):
    email: str = Field(min_length=3, max_length=200, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    name: str | None = Field(default=None, max_length=200)
    admin: bool = False


@app.post("/internal/admin/onboard")
async def admin_onboard(
    body: OnboardRequest,
    x_fred_live_token: str | None = Header(default=None),
    token: str | None = Query(default=None),
):
    """Run the full server-side onboarding (Vexa user + key + webhook + roster)
    on behalf of the web app's admin approval flow — the web container cannot
    reach Vexa's admin API (bound to host loopback), the worker can. The
    per-user API key is stored on disk and deliberately NOT returned."""
    _require_token(x_fred_live_token, token)
    import asyncio

    import httpx as _httpx

    from .onboard import onboard_user

    try:
        result = await asyncio.to_thread(
            onboard_user, settings, body.email, name=body.name, admin=body.admin
        )
    except _httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Vexa API refused: {exc.response.status_code} {exc.response.text[:200]}",
        )
    except (_httpx.HTTPError, SystemExit) as exc:
        raise HTTPException(status_code=502, detail=f"Onboarding failed: {exc}")
    logger.info("Onboarded %s as Vexa user %s (admin=%s)", body.email, result["user_id"], body.admin)
    return {"user_id": result["user_id"], "created": result["created"]}


@app.post("/internal/admin/offboard")
async def admin_offboard(
    body: OnboardRequest,
    x_fred_live_token: str | None = Header(default=None),
    token: str | None = Query(default=None),
):
    """Drop a user from the roster + key store (their session, capture token
    and autojoin launches stop on the next request). The Vexa user and their
    meeting history are left intact."""
    _require_token(x_fred_live_token, token)
    from .onboard import offboard_user

    result = offboard_user(settings, body.email)
    logger.info("Offboarded %s: %s", body.email, result)
    return result


class RenameSpeakersRequest(BaseModel):
    renames: dict[str, str] = Field(min_length=1, max_length=30)


@app.post("/internal/meetings/{meeting_id}/rename-speakers")
async def rename_meeting_speakers(
    meeting_id: str,
    body: RenameSpeakersRequest,
    x_fred_live_token: str | None = Header(default=None),
    token: str | None = Query(default=None),
):
    """Batch-relabel speakers on the newest completed run (hybrid-room fix:
    shared-mic voices get the account owner's name or stay SPEAKER_N)."""
    _require_token(x_fred_live_token, token)
    if not re.fullmatch(r"[\w-]+", meeting_id):
        raise HTTPException(status_code=404, detail=f"No runs found for meeting {meeting_id}")
    from .speaker_rename import rename_speakers

    try:
        outcome = rename_speakers(settings.meetings_dir, meeting_id, body.renames)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return outcome


class AskRequest(BaseModel):
    question: str
    history: list[dict[str, str]] = Field(default_factory=list, max_length=20)


@app.post("/internal/meetings/{meeting_id}/ask")
async def ask_meeting(
    meeting_id: str,
    request: AskRequest,
    x_fred_live_token: str | None = Header(default=None),
    token: str | None = Query(default=None),
):
    _require_token(x_fred_live_token, token)
    if not re.fullmatch(r"[\w-]+", meeting_id):
        raise HTTPException(status_code=404, detail=f"No meeting {meeting_id}")

    meeting_dir = settings.meetings_dir / meeting_id
    run_dirs = sorted(
        (d for d in meeting_dir.iterdir() if d.is_dir() and (d / "transcript.json").exists()),
        reverse=True,
    ) if meeting_dir.is_dir() else []
    if not run_dirs:
        raise HTTPException(status_code=404, detail=f"No transcript for meeting {meeting_id}")
    run_dir = run_dirs[0]

    transcript = TranscriptResult.model_validate(
        json.loads((run_dir / "transcript.json").read_text(encoding="utf-8"))
    )
    # speech, then the meeting chat (typed messages: links, numbers) as its own block
    transcript_text = transcript_for_llm(transcript).strip()

    insights_block = ""
    result_path = run_dir / "result.json"
    latest_path = meeting_dir / "latest.json"
    for candidate in (result_path, latest_path):
        if candidate.exists():
            insights = json.loads(candidate.read_text(encoding="utf-8")).get("insights")
            if insights:
                insights_block = f"\n\nStructured meeting notes:\n{json.dumps(insights, ensure_ascii=False)}"
                break

    system_prompt = (
        "You answer questions about ONE meeting, using only its transcript and notes below. "
        "If the answer is not in the meeting, say so plainly. Answer in the language the "
        f"question was asked in. {processor.llm.glossary()}\n\n"
        f"Transcript:\n{transcript_text}{insights_block}"
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(m for m in request.history if m.get("role") in ("user", "assistant"))
    messages.append({"role": "user", "content": request.question})

    answer = await processor.llm.chat(messages)
    return {"meeting_id": meeting_id, "run": run_dir.name, "answer": answer}


# Vexa's outbound-events ledger can deliver the same webhook more than once;
# remember recent event_ids so a redelivery doesn't re-run the whole GPU pipeline.
_SEEN_EVENT_IDS: dict[str, float] = {}
_SEEN_EVENT_TTL_SECONDS = 3600


def _is_duplicate_event(event_id: str | None) -> bool:
    if not event_id:
        return False
    import time

    now = time.monotonic()
    for key, seen_at in list(_SEEN_EVENT_IDS.items()):
        if now - seen_at > _SEEN_EVENT_TTL_SECONDS:
            del _SEEN_EVENT_IDS[key]
    if event_id in _SEEN_EVENT_IDS:
        return True
    _SEEN_EVENT_IDS[event_id] = now
    return False


# Vexa also emits meeting.completed TWICE per meeting with DISTINCT event_ids
# (once from the completion handler, once from the recording finalizer ~0.5s
# later), so event_id dedup alone still double-processes. Each sender stamps
# its own end_time (observed 313ms apart), so the meeting id is the ONLY
# stable key — any repeat completion of the same meeting inside the window is
# a duplicate. The window is deliberately small so deliberate replays of a
# saved hook (scripts/replay_hook.sh, typically minutes later) still process.
_SEEN_COMPLETIONS: dict[str, float] = {}
_SEEN_COMPLETION_WINDOW_SECONDS = 120


def _is_duplicate_completion(meeting_id: int | str) -> bool:
    import time

    key = str(meeting_id)
    now = time.monotonic()
    for k, seen_at in list(_SEEN_COMPLETIONS.items()):
        if now - seen_at > _SEEN_COMPLETION_WINDOW_SECONDS:
            del _SEEN_COMPLETIONS[k]
    if key in _SEEN_COMPLETIONS:
        return True
    _SEEN_COMPLETIONS[key] = now
    return False


# Vexa signs deliveries with HMAC-SHA256 over "{timestamp}.{body}" using the
# secret registered at PUT /user/webhook (X-Webhook-Signature: sha256=<hex>,
# X-Webhook-Timestamp: <unix>). When FRED_LIVE_WEBHOOK_SECRET is set, the hook
# endpoint REQUIRES a valid, fresh signature — the ?token= alone no longer
# admits meeting-completed posts (replay_hook.sh signs with the same secret).
_WEBHOOK_TIMESTAMP_TOLERANCE_SECONDS = 300


def _verify_webhook_signature(
    raw_body: bytes, signature_header: str | None, timestamp_header: str | None
) -> tuple[bool, str]:
    import hashlib
    import hmac as hmac_mod
    import time

    secret = settings.webhook_secret
    if secret is None or not secret.get_secret_value().strip():
        return True, "no secret configured"
    if not signature_header or not timestamp_header:
        return False, "missing signature headers"
    try:
        ts = int(timestamp_header)
    except ValueError:
        return False, "bad timestamp"
    if abs(time.time() - ts) > _WEBHOOK_TIMESTAMP_TOLERANCE_SECONDS:
        return False, "stale timestamp"
    if not signature_header.startswith("sha256="):
        return False, "bad signature format"
    expected = hmac_mod.new(
        secret.get_secret_value().strip().encode(),
        f"{ts}.".encode() + raw_body,
        hashlib.sha256,
    ).hexdigest()
    if not hmac_mod.compare_digest(signature_header[len("sha256="):], expected):
        return False, "signature mismatch"
    return True, "ok"


def _normalize_hook_payload(body: dict) -> MeetingCompletedPayload:
    """Accept both hook shapes:
    - new enveloped per-user webhook: {event_id, event_type, api_version, created_at,
      data: {meeting: {...}}} (Vexa >= 0.10.6, webhook_delivery.build_envelope)
    - legacy POST_MEETING_HOOKS: {event, meeting: {...}} (kept for replaying saved hooks)
    """
    if "event_type" in body:
        meeting = (body.get("data") or {}).get("meeting")
        if not isinstance(meeting, dict):
            raise HTTPException(status_code=422, detail="Envelope missing data.meeting")
        return MeetingCompletedPayload(event=body["event_type"], meeting=meeting)
    return MeetingCompletedPayload.model_validate(body)


@app.post("/internal/vexa/meeting-completed", status_code=202)
async def meeting_completed(
    request: Request,
    background_tasks: BackgroundTasks,
    x_fred_live_token: str | None = Header(default=None),
    token: str | None = Query(default=None),
    x_webhook_signature: str | None = Header(default=None),
    x_webhook_timestamp: str | None = Header(default=None),
) -> dict[str, str | int]:
    _require_token(x_fred_live_token, token)
    raw_body = await request.body()
    ok, reason = _verify_webhook_signature(raw_body, x_webhook_signature, x_webhook_timestamp)
    if not ok:
        logger.warning("Rejected meeting-completed hook: %s", reason)
        raise HTTPException(status_code=401, detail=f"Webhook signature invalid: {reason}")
    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="Body is not valid JSON") from exc
    if _is_duplicate_event(body.get("event_id")):
        return {"status": "duplicate", "event_id": body["event_id"]}
    payload = _normalize_hook_payload(body)
    if payload.event != "meeting.completed":
        # The per-user webhook also delivers meeting.status_change etc.; a 4xx here
        # would only trigger the sender's retry/backoff for events we never consume.
        return {"status": "ignored", "event": payload.event}
    if _is_duplicate_completion(payload.meeting.id):
        return {"status": "duplicate", "meeting_id": payload.meeting.id}

    background_tasks.add_task(_process_in_background, payload)
    return {"status": "accepted", "meeting_id": payload.meeting.id}


# --- voiceprint management (proxied by the web app's admin/self-enroll UI) ---

@app.get("/internal/voiceprints")
async def list_voiceprints(
    x_fred_live_token: str | None = Header(default=None),
    token: str | None = Query(default=None),
):
    _require_token(x_fred_live_token, token)
    from .voiceprints import VoiceprintStore

    store = VoiceprintStore(settings.data_dir)
    return {
        "prints": [
            {
                "name": p.name,
                "slug": store.slug_for(p.name),
                "seconds": p.seconds,
                "samples": p.samples,
                "model": p.model,
                "metadata": p.metadata or {},
            }
            for p in store.load_all()
        ]
    }


@app.post("/internal/voiceprints/enroll", status_code=201)
async def enroll_voiceprint(
    request: Request,
    x_fred_live_token: str | None = Header(default=None),
    token: str | None = Query(default=None),
):
    """Self-enrollment endpoint (multipart: name, email, source, notice_version,
    audio). The raw audio only ever exists as a temp file for the seconds it
    takes to compute the embedding; it is deleted before this returns (GDPR
    data minimization; the 192-float embedding cannot reconstruct speech)."""
    import asyncio
    import tempfile

    _require_token(x_fred_live_token, token)
    form = await request.form()
    name = str(form.get("name") or "").strip()[:80]
    audio = form.get("audio")
    if not name or audio is None or not hasattr(audio, "read"):
        raise HTTPException(status_code=422, detail="name and audio file are required")
    metadata = {
        "email": str(form.get("email") or "").strip().lower()[:120],
        "source": str(form.get("source") or "upload")[:30],
        "notice_version": str(form.get("notice_version") or "")[:20],
        "consented_at": int(__import__("time").time()),
    }
    payload = await audio.read()
    if len(payload) > 30 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio too large (30MB max)")
    if len(payload) < 1000:
        raise HTTPException(status_code=422, detail="Audio too short")

    from .audio_io import SAMPLE_RATE, load_waveform_16k
    from .voiceprints import MIN_ENROLL_SECONDS, VoiceprintStore, enroll

    def _do_enroll():
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=True) as tmp:
            tmp.write(payload)
            tmp.flush()
            waveform = load_waveform_16k(Path(tmp.name))
            # temp file removed by the context manager; only the waveform
            # (and then the embedding) survive
            store = VoiceprintStore(settings.data_dir)
            backend = processor.get_voiceprint_backend()
            return enroll(store, backend, name, [waveform], metadata=metadata), len(waveform) / SAMPLE_RATE

    try:
        print_, seconds = await asyncio.to_thread(_do_enroll)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        logger.exception("Voiceprint enrollment failed for %s", name)
        raise HTTPException(status_code=500, detail="Enrollment failed; check worker logs")
    return {
        "name": print_.name,
        "slug": VoiceprintStore(settings.data_dir).slug_for(print_.name),
        "seconds": round(print_.seconds, 1),
        "model": print_.model,
        "warning": (
            f"Only {seconds:.0f}s of audio; identification improves with "
            f"{int(MIN_ENROLL_SECONDS)}+ seconds" if seconds < MIN_ENROLL_SECONDS else None
        ),
    }


@app.delete("/internal/voiceprints/{slug}")
async def delete_voiceprint(
    slug: str,
    x_fred_live_token: str | None = Header(default=None),
    token: str | None = Query(default=None),
):
    _require_token(x_fred_live_token, token)
    from .voiceprints import VoiceprintStore

    if not VoiceprintStore(settings.data_dir).delete_slug(slug):
        raise HTTPException(status_code=404, detail="No such voiceprint")
    logger.info("Voiceprint %s deleted via API", slug)
    return {"status": "deleted", "slug": slug}


class VoiceprintAccountRequest(BaseModel):
    # None or "" unlinks.
    email: str | None = Field(default=None, max_length=120)
    by: str = Field(default="", max_length=120)


@app.post("/internal/voiceprints/{slug}/account")
async def link_voiceprint_account(
    slug: str,
    body: VoiceprintAccountRequest,
    x_fred_live_token: str | None = Header(default=None),
    token: str | None = Query(default=None),
):
    """Link a print to a Fred account (admin page). Prints from the operator
    CLI carry no email; without one, Fred cannot share a meeting with that
    person automatically. The web app checks that the email is on the roster."""
    _require_token(x_fred_live_token, token)
    from .voiceprints import VoiceprintStore

    metadata = VoiceprintStore(settings.data_dir).set_account(slug, body.email, body.by)
    if metadata is None:
        raise HTTPException(status_code=404, detail="No such voiceprint")
    logger.info("Voiceprint %s %s by %s", slug,
                f"linked to {metadata['email']}" if metadata.get("email") else "unlinked", body.by or "?")
    return {"slug": slug, "metadata": metadata}
