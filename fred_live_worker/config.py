from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


# Defaults are RELATIVE to the checkout (mirrors fred_live_web/config.py):
# absolute paths baked in here made the worker unstartable on any other
# machine without editing source. Every one is still overridable by env.
REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_WHISPER_DIR = str(REPO_ROOT / "models" / "openai__whisper-large-v3")
DEFAULT_GGUF_PATH = str(
    REPO_ROOT / "models" / "unsloth__Qwen3.8-27B-GGUF" / "Qwen3.8-27B-UD-Q6_K.gguf"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.worker",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_host: str = Field(default="0.0.0.0", alias="FRED_LIVE_APP_HOST")
    app_port: int = Field(default=8088, alias="FRED_LIVE_APP_PORT")
    auth_token: SecretStr | None = Field(default=None, alias="FRED_LIVE_AUTH_TOKEN")
    data_dir: Path = Field(default=REPO_ROOT / "data", alias="FRED_LIVE_DATA_DIR")
    # The team's own names, for the notes and Ask (llm.glossary_instruction). Default:
    # glossary.txt in the data directory; no file, no glossary.
    glossary_file: Path | None = Field(default=None, alias="FRED_LIVE_GLOSSARY_FILE")
    recordings_dir: Path = Field(
        default=REPO_ROOT / "recordings",
        alias="FRED_LIVE_RECORDINGS_DIR",
    )
    asr_model_dir: Path = Field(default=Path(DEFAULT_WHISPER_DIR), alias="FRED_LIVE_ASR_MODEL_DIR")
    # "whispercpp" (default; GGML, ~4GB, seconds per meeting) or "transformers"
    # (HF pipeline; needed for .bin-only fine-tunes, such as a Catalan one).
    asr_backend: str = Field(default="whispercpp", alias="FRED_LIVE_ASR_BACKEND")
    # Anti-hallucination: retry a span with temperature sampling when the
    # greedy decode produces a repetition loop (boundary-sensitive decoder
    # instability observed in the field).
    asr_loop_retry: bool = Field(default=True, alias="FRED_LIVE_ASR_LOOP_RETRY")
    # Spans with less speech-energy than this are skipped before decoding
    # (Whisper fabricates stock phrases on near-silent slivers). 0 disables.
    asr_min_span_speech_seconds: float = Field(
        default=0.3, alias="FRED_LIVE_ASR_MIN_SPAN_SPEECH_SECONDS"
    )
    # That gate's energy threshold follows the recording: this many dB below its
    # loud speech, never above the fixed 0.01 RMS. A quiet microphone otherwise
    # loses whole turns. 0 keeps the fixed threshold.
    asr_span_gate_relative_db: float = Field(
        default=30.0, alias="FRED_LIVE_ASR_SPAN_GATE_RELATIVE_DB"
    )
    # Second language tag in Whisper's prompt (pipeline._second_language_pass):
    # spans outside the meeting's main language get the main one as a second tag,
    # and spans in the main language get the meeting's second language, only when
    # that language holds at least this share of the speech. Needs whisper-batch.
    asr_second_language: bool = Field(default=True, alias="FRED_LIVE_ASR_SECOND_LANGUAGE")
    asr_second_language_min_share: float = Field(
        default=0.10, alias="FRED_LIVE_ASR_SECOND_LANGUAGE_MIN_SHARE"
    )
    # A language let into the inventory by cumulative evidence alone must also be
    # heard by the voice model (ECAPA), or its spans are decoded with the main
    # language first (pipeline._unheard_languages). Needs whisper-batch.
    asr_confirm_cumulative_languages: bool = Field(
        default=True, alias="FRED_LIVE_ASR_CONFIRM_CUMULATIVE_LANGUAGES"
    )
    # A language a long span let in that the voice model does not hear stays the
    # language of whoever spoke it at length; short spans of the other speakers read
    # in it are decoded with their own language first. Needs whisper-batch.
    asr_speaker_languages: bool = Field(default=True, alias="FRED_LIVE_ASR_SPEAKER_LANGUAGES")
    # Known-speaker identification from enrolled voiceprints
    # (python -m fred_live_worker.enroll). No-op until someone is enrolled.
    voiceprints_enabled: bool = Field(default=True, alias="FRED_LIVE_VOICEPRINTS_ENABLED")
    # "auto" = ReDimNet2 (wespeaker) when installed, else speechbrain ECAPA.
    voiceprint_backend: str = Field(default="auto", alias="FRED_LIVE_VOICEPRINT_BACKEND")
    voiceprint_model_dir: str = Field(default="", alias="FRED_LIVE_VOICEPRINT_MODEL_DIR")
    # Cosine gates for assigning a name to a cluster: best >= threshold AND
    # best - runner_up >= margin. Calibrated on real meetings: genuine speakers
    # score 0.58-0.91 (cross-microphone low end 0.63 after multi-mic
    # enrollment), and the margin gate is the second lock.
    voiceprint_threshold: float = Field(default=0.55, alias="FRED_LIVE_VOICEPRINT_THRESHOLD")
    voiceprint_margin: float = Field(default=0.08, alias="FRED_LIVE_VOICEPRINT_MARGIN")
    # The threshold for someone with no tile on the page, who is not the meeting's
    # owner either: there the voice is the only evidence. An unenrolled guest scored
    # 0.607 and 0.621 against a colleague who was not in the meeting (2026-11-04);
    # on 22 meetings this unnames 2 of 32 real people heard without a tile.
    voiceprint_off_page_threshold: float = Field(
        default=0.65, alias="FRED_LIVE_VOICEPRINT_OFF_PAGE_THRESHOLD"
    )
    # A gate-passing voice match REPLACES a conflicting DOM-event name.
    # Why: in hybrid rooms every voice arrives
    # through the mic owner's Meet identity, so DOM attribution is
    # systematically wrong for everyone else in the room.
    voiceprint_override_dom: bool = Field(
        default=True, alias="FRED_LIVE_VOICEPRINT_OVERRIDE_DOM"
    )
    # Extension recordings keep the microphone and the tab on separate channels; the
    # worker sums them unless the tab repeats the microphone (channels.py).
    audio_channels_enabled: bool = Field(default=True, alias="FRED_LIVE_AUDIO_CHANNELS")
    # Turns diarization put in someone else's cluster go back to their owner when
    # that person's tile was lit and the voice is theirs (needs voiceprints).
    page_voices_enabled: bool = Field(default=True, alias="FRED_LIVE_PAGE_VOICES")
    # And any turn whose voice alone is someone else in the meeting (needs voiceprints).
    voice_sweep_enabled: bool = Field(default=True, alias="FRED_LIVE_VOICE_SWEEP")
    # Automatic sharing: each meeting is shared,
    # read only, with the registered users recognized BY VOICE in it; on by
    # default, every owner can turn it off in Settings. Stricter than naming a
    # speaker: of the 144 matches accepted since 2026-10-01 (0.590-0.934,
    # median 0.826) only the weakest falls below 0.60, and the highest
    # rejected score was 0.532. The speech minimum keeps a short, noisy
    # cluster from opening a meeting to anyone.
    auto_share_enabled: bool = Field(default=True, alias="FRED_LIVE_AUTO_SHARE")
    auto_share_min_score: float = Field(default=0.60, alias="FRED_LIVE_AUTO_SHARE_MIN_SCORE")
    auto_share_min_seconds: float = Field(default=10.0, alias="FRED_LIVE_AUTO_SHARE_MIN_SECONDS")
    # And for someone with no tile on the page: a name can be wrong and the owner
    # renames it, a share opens the meeting. Above the off-page naming threshold,
    # since the guest above reached 0.621.
    auto_share_off_page_min_score: float = Field(
        default=0.70, alias="FRED_LIVE_AUTO_SHARE_OFF_PAGE_MIN_SCORE"
    )
    whispercpp_bin: Path = Field(
        default=REPO_ROOT / "vendor" / "whisper.cpp" / "build" / "bin" / "whisper-cli",
        alias="FRED_LIVE_WHISPERCPP_BIN",
    )
    whispercpp_model: Path = Field(
        default=REPO_ROOT / "models" / "ggml-whisper" / "ggml-large-v3.bin",
        alias="FRED_LIVE_WHISPERCPP_MODEL",
    )
    # whisper-batch (tools/whisper-batch, built by scripts/build_whisper_batch.sh)
    # keeps the model loaded across a meeting's span calls instead of one
    # whisper-cli per call (~0.9 s of model load each, hundreds of calls). Same
    # output, call for call; whisper-cli runs the calls when it is absent or fails.
    asr_whispercpp_batch: bool = Field(default=True, alias="FRED_LIVE_ASR_WHISPERCPP_BATCH")
    # Default: whisper-batch next to FRED_LIVE_WHISPERCPP_BIN.
    whispercpp_batch_bin: Path | None = Field(default=None, alias="FRED_LIVE_WHISPERCPP_BATCH_BIN")
    asr_language: str = Field(default="auto", alias="FRED_LIVE_ASR_LANGUAGE")
    diarization_enabled: bool = Field(default=True, alias="FRED_LIVE_DIARIZATION_ENABLED")
    # pyannote embeds every (10 s chunk, local speaker) pair, including speakers
    # silent for the whole chunk, whose embedding is one constant vector: 42-46%
    # of the diarization time on real meetings. Skipping them left the turns
    # identical on every meeting compared (diarization.py).
    diarization_skip_silent: bool = Field(default=True, alias="FRED_LIVE_DIARIZATION_SKIP_SILENT")
    # Second diarizer: who speaks comes from Nemotron 3 Diarization and the speech
    # it missed from pyannote (speaker_merge.fuse_turns). Needs
    # scripts/setup_nemotron.sh; without it the worker logs and keeps pyannote.
    diarization_nemotron: bool = Field(default=False, alias="FRED_LIVE_DIARIZATION_NEMOTRON")
    nemotron_python: Path = Field(
        default=REPO_ROOT / ".venv-nemotron" / "bin" / "python", alias="FRED_LIVE_NEMOTRON_PYTHON"
    )
    nemotron_model_dir: Path = Field(
        default=REPO_ROOT / "models" / "nvidia__Nemotron-3-Diarization",
        alias="FRED_LIVE_NEMOTRON_MODEL_DIR",
    )
    nemotron_threshold: float = Field(default=0.35, alias="FRED_LIVE_NEMOTRON_THRESHOLD")
    # Transcribe each speaker-turn block with its own language detection, so
    # code-switched meetings come out in the languages actually spoken instead of
    # being pinned to the language of the first 30 seconds. Requires diarization.
    asr_multilingual: bool = Field(default=True, alias="FRED_LIVE_ASR_MULTILINGUAL")
    hf_token: SecretStr | None = Field(default=None, alias="FRED_LIVE_HF_TOKEN")
    asr_chunk_length_s: int = Field(default=30, alias="FRED_LIVE_ASR_CHUNK_LENGTH_S")
    asr_batch_size: int = Field(default=8, alias="FRED_LIVE_ASR_BATCH_SIZE")
    llm_base_url: str = Field(default="http://127.0.0.1:8081", alias="FRED_LIVE_LLM_BASE_URL")
    llm_model: str = Field(default="qwen3.8-27b", alias="FRED_LIVE_LLM_MODEL")
    llm_api_key: SecretStr | None = Field(default=None, alias="FRED_LIVE_LLM_API_KEY")
    llm_timeout_seconds: int = Field(default=600, alias="FRED_LIVE_LLM_TIMEOUT_SECONDS")
    llm_temperature: float = Field(default=0.1, alias="FRED_LIVE_LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=2048, alias="FRED_LIVE_LLM_MAX_TOKENS")
    llm_ctx_size: int = Field(default=65536, alias="FRED_LIVE_LLM_CTX_SIZE")
    llm_output_language: str = Field(default="same-as-transcript", alias="FRED_LIVE_LLM_OUTPUT_LANGUAGE")
    # Notes in two steps: decisions and tasks listed part by part, then the notes
    # written from the whole transcript and that list. One model call per ~12k
    # characters of transcript, plus one. false = one call, as before.
    llm_two_step: bool = Field(default=True, alias="FRED_LIVE_LLM_TWO_STEP")
    # true = the call that writes the notes reasons first; every other call
    # follows llm_disable_thinking. Off by default: measured on AMI it found as
    # much, with 2-3 points fewer items backed by the transcript and ~25% more
    # words, in a tenth of the time, and it never runs away.
    llm_notes_thinking: bool = Field(default=False, alias="FRED_LIVE_LLM_NOTES_THINKING")
    llm_disable_thinking: bool = Field(default=True, alias="FRED_LIVE_LLM_DISABLE_THINKING")
    llm_gguf_path: Path = Field(default=Path(DEFAULT_GGUF_PATH), alias="FRED_LIVE_LLM_GGUF_PATH")
    vexa_api_base_url: str | None = Field(default=None, alias="VEXA_API_BASE_URL")
    vexa_api_key: SecretStr | None = Field(default=None, alias="VEXA_API_KEY")
    vexa_timeout_seconds: int = Field(default=120, alias="VEXA_TIMEOUT_SECONDS")
    prefer_local_recordings: bool = Field(default=True, alias="FRED_LIVE_PREFER_LOCAL_RECORDINGS")
    recording_settle_seconds: int = Field(default=120, alias="FRED_LIVE_RECORDING_SETTLE_SECONDS")
    gcal_credentials: Path | None = Field(default=None, alias="FRED_LIVE_GCAL_CREDENTIALS")
    autojoin_lead_seconds: int = Field(default=90, alias="FRED_LIVE_AUTOJOIN_LEAD_SECONDS")
    autojoin_poll_seconds: int = Field(default=60, alias="FRED_LIVE_AUTOJOIN_POLL_SECONDS")
    autojoin_bot_name: str = Field(default="Fred Notetaker", alias="FRED_LIVE_AUTOJOIN_BOT_NAME")
    # Launch bots signed in as the bot's Google account (requires the one-time
    # browser_session login; see docs/INSTALL.md, section 11).
    autojoin_authenticated: bool = Field(default=True, alias="FRED_LIVE_AUTOJOIN_AUTHENTICATED")
    # Add the bot's Google account as an attendee to events before joining.
    # An invited signed-in account joins without the "possibly risky" label
    # (relationship-based trust — the industry-standard fix, per Recall.ai docs).
    autojoin_invite_bot: bool = Field(default=True, alias="FRED_LIVE_AUTOJOIN_INVITE_BOT")
    bot_google_account: str | None = Field(default=None, alias="FRED_LIVE_BOT_GOOGLE_ACCOUNT")
    # How long past the scheduled start the bot waits for the first human before
    # giving up. Vexa's default no_one_joined_timeout is 120s — the autojoin lead
    # already burns most of that before the event starts, so we pass an explicit
    # override of (lead + grace) at launch.
    autojoin_no_show_grace_seconds: int = Field(
        default=600, alias="FRED_LIVE_AUTOJOIN_NO_SHOW_GRACE_SECONDS"
    )
    # How long the bot stays after everyone else has left. Guests can't "end
    # for everyone", so when the last human leaves, this is what actually ends
    # the meeting — Vexa's default is 15 min, which delays the summary by 15
    # minutes of silent recording.
    autojoin_alone_timeout_seconds: int = Field(
        default=180, alias="FRED_LIVE_AUTOJOIN_ALONE_TIMEOUT_SECONDS"
    )
    # Shared with the web app (same .env.worker): signs per-user capture
    # tokens for the browser extension.
    web_session_secret: str | None = Field(default=None, alias="FRED_LIVE_WEB_SESSION_SECRET")
    # Same allowlist the web app uses ("email:vexa_user_id[:admin],..."). When
    # set, capture tokens are re-checked against it on every request, so
    # removing an email revokes that user's long-lived capture token too (not
    # just their web session). Empty = capture tokens gated by signature only.
    web_users: str = Field(default="", alias="FRED_LIVE_WEB_USERS")
    # Hard ceiling on one extension capture's total uploaded audio, summed
    # across parts. A valid token holder can otherwise fill the disk between
    # daily retention sweeps. 0 disables.
    capture_max_session_mb: int = Field(
        default=2048, alias="FRED_LIVE_CAPTURE_MAX_SESSION_MB"
    )
    # Extension captures of the same meet code by the same user within this
    # gap resume the SAME meeting (multi-part recording) instead of creating a
    # new one; big enough for a pause, small enough that a recurring meeting's
    # reused link next day/week stays a separate meeting. 0 disables.
    capture_resume_gap_minutes: int = Field(
        default=30, alias="FRED_LIVE_CAPTURE_RESUME_GAP_MINUTES"
    )
    # Vexa per-user webhook HMAC secret (registered via PUT /user/webhook).
    # When set, /internal/vexa/meeting-completed REQUIRES a valid
    # X-Webhook-Signature (sha256 over "{ts}.{body}") with a fresh timestamp.
    webhook_secret: SecretStr | None = Field(default=None, alias="FRED_LIVE_WEBHOOK_SECRET")
    # Delete recordings (audio only — transcripts and summaries are kept) older
    # than this many days. 0 disables. Deletion runs through the meeting-api
    # container because Vexa writes recordings as uid 999.
    retention_days: int = Field(default=15, alias="FRED_LIVE_RETENTION_DAYS")
    retention_container: str = Field(
        default="vexa-meeting-api-1", alias="FRED_LIVE_RETENTION_CONTAINER"
    )
    # Periodically look for audio on disk that produced NO run at all (a
    # capture whose queued processing died with the worker, a webhook Vexa
    # could not deliver). Recoverable captures are reprocessed automatically;
    # everything else lands in data/orphaned_recordings.json.
    orphan_sweep_enabled: bool = Field(default=True, alias="FRED_LIVE_ORPHAN_SWEEP_ENABLED")
    # How quiet a recording must be before it counts as orphaned; must exceed
    # the longest normal ASR+LLM run so a meeting being processed right now is
    # never swept.
    orphan_min_age_minutes: float = Field(
        default=120.0, alias="FRED_LIVE_ORPHAN_MIN_AGE_MINUTES"
    )
    # Skip ASR/LLM when the recording has less than this many seconds of speech-
    # energy audio; the run is persisted with quality="no_speech". Prevents
    # Whisper hallucination summaries for bot-alone meetings. 0 disables.
    min_active_speech_seconds: float = Field(
        default=5.0, alias="FRED_LIVE_MIN_ACTIVE_SPEECH_SECONDS"
    )
    # Intra-block language-switch detection: same-speaker blocks are split at
    # pauses, each chunk gets a fast whisper.cpp -dl language ID (<1s), adjacent
    # same-language chunks re-merge, and each span is transcribed with its
    # language pinned. Without this, ONE language is detected per block and a
    # mid-block switch gets translated into the block's language.
    asr_lid_split_enabled: bool = Field(default=True, alias="FRED_LIVE_ASR_LID_SPLIT")
    # Blocks shorter than this keep single-pass auto-detection (LID overhead
    # isn't worth it and short chunks detect poorly).
    asr_lid_min_block_seconds: float = Field(
        default=8.0, alias="FRED_LIVE_ASR_LID_MIN_BLOCK_SECONDS"
    )
    asr_lid_min_pause_seconds: float = Field(
        default=0.6, alias="FRED_LIVE_ASR_LID_MIN_PAUSE_SECONDS"
    )
    asr_lid_min_chunk_seconds: float = Field(
        default=3.0, alias="FRED_LIVE_ASR_LID_MIN_CHUNK_SECONDS"
    )
    # Short-span language rescoring (ECAPA/VoxLingua107): Whisper's LID is near
    # chance on sub-8s code-switched chunks (hallucinated sv/pl/pt in an EN/TR
    # meeting). Spans >= evidence_seconds establish the meeting's language
    # inventory; shorter spans are decided by ECAPA probabilities constrained
    # to that inventory plus a same-speaker stickiness prior.
    lid_rescore_enabled: bool = Field(default=True, alias="FRED_LIVE_LID_RESCORE")
    asr_lang_evidence_seconds: float = Field(
        default=8.0, alias="FRED_LIVE_ASR_LANG_EVIDENCE_SECONDS"
    )
    lid_model_dir: Path = Field(
        default=REPO_ROOT / "models" / "speechbrain__lang-id-voxlingua107-ecapa",
        alias="FRED_LIVE_LID_MODEL_DIR",
    )
    # Code-switching escape valve: a short span may introduce a language
    # outside the inventory iff ECAPA is this confident AND the span is this
    # long (both gates needed — a 1.7s hallucination once scored sv=0.88).
    lid_escape_min_prob: float = Field(default=0.85, alias="FRED_LIVE_LID_ESCAPE_MIN_PROB")
    lid_escape_min_seconds: float = Field(default=3.0, alias="FRED_LIVE_LID_ESCAPE_MIN_SECONDS")

    @property
    def meetings_dir(self) -> Path:
        return self.data_dir / "meetings"

    @property
    def glossary_path(self) -> Path:
        return self.glossary_file or self.data_dir / "glossary.txt"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

