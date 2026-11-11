from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class VexaMediaFile(BaseModel):
    id: int | str
    type: str | None = None
    format: str | None = None
    storage_path: str | None = None
    storage_backend: str | None = None
    file_size_bytes: int | None = None
    duration_seconds: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


class VexaRecording(BaseModel):
    id: int | str
    meeting_id: int | str | None = None
    user_id: int | str | None = None
    session_uid: str | None = None
    source: str | None = None
    status: str | None = None
    error_message: str | None = None
    created_at: str | None = None
    completed_at: str | None = None
    media_files: list[VexaMediaFile] = Field(default_factory=list)


class VexaMeeting(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int | str
    user_id: int | str
    user_email: str | None = None
    platform: str | None = None
    status: str | None = None
    duration_seconds: float | None = None
    start_time: str | None = None
    end_time: str | None = None
    created_at: str | None = None
    transcription_enabled: bool | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class MeetingCompletedPayload(BaseModel):
    event: str
    meeting: VexaMeeting


class TranscriptSegment(BaseModel):
    start_seconds: float | None = None
    end_seconds: float | None = None
    text: str
    speaker: str | None = None
    language: str | None = None


class ChatMessage(BaseModel):
    """A message typed in the meeting's own chat, read from the meeting page by
    the browser extension. Kept apart from the spoken segments on purpose: it
    never went through ASR, so none of the speech filters apply to it, and a
    sender who only typed must not count as a voice anywhere."""

    sender: str
    text: str
    offset_seconds: float | None = None
    links: list[str] = Field(default_factory=list)


class VoiceMatch(BaseModel):
    """An enrolled voice recognized in the meeting (gate-passing voiceprint
    match), with the print's account email when it has one. Drives automatic
    sharing; `seconds` is the speech attributed to that cluster."""

    speaker: str
    name: str
    email: str | None = None
    score: float
    seconds: float = 0.0


class TranscriptResult(BaseModel):
    text: str
    language: str | None = None
    segments: list[TranscriptSegment] = Field(default_factory=list)
    # Maps diarization labels (SPEAKER_00, ...) to real participant names where known.
    speaker_names: dict[str, str] = Field(default_factory=dict)
    chat_messages: list[ChatMessage] = Field(default_factory=list)
    voice_matches: list[VoiceMatch] = Field(default_factory=list)


class ActionItem(BaseModel):
    task: str
    owner: str | None = None
    due_date: str | None = None
    notes: str | None = None


class MeetingInsights(BaseModel):
    summary: str
    most_important_points: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    risks_or_open_questions: list[str] = Field(default_factory=list)

