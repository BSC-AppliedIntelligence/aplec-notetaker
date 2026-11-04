from __future__ import annotations

from pathlib import Path

import httpx

from .config import Settings
from .schemas import VexaMediaFile, VexaRecording


class RecordingLookupError(RuntimeError):
    pass


class VexaClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _headers(self) -> dict[str, str]:
        if not self.settings.vexa_api_key:
            raise RecordingLookupError("VEXA_API_KEY is not configured")
        return {"X-API-Key": self.settings.vexa_api_key.get_secret_value()}

    def _base_url(self) -> str:
        if not self.settings.vexa_api_base_url:
            raise RecordingLookupError("VEXA_API_BASE_URL is not configured")
        return self.settings.vexa_api_base_url.rstrip("/")

    async def list_recordings(self, meeting_id: int | str) -> list[VexaRecording]:
        async with httpx.AsyncClient(timeout=self.settings.vexa_timeout_seconds) as client:
            response = await client.get(
                f"{self._base_url()}/recordings",
                headers=self._headers(),
                params={"meeting_id": meeting_id, "limit": 100, "offset": 0},
            )
            response.raise_for_status()
            payload = response.json()
        recordings = payload.get("recordings", [])
        return [VexaRecording.model_validate(recording) for recording in recordings]

    async def download_media_file(
        self,
        recording: VexaRecording,
        media_file: VexaMediaFile,
        output_path: Path,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient(timeout=self.settings.vexa_timeout_seconds) as client:
            async with client.stream(
                "GET",
                f"{self._base_url()}/recordings/{recording.id}/media/{media_file.id}/raw",
                headers=self._headers(),
            ) as response:
                response.raise_for_status()
                with output_path.open("wb") as handle:
                    async for chunk in response.aiter_bytes():
                        handle.write(chunk)

        return output_path

