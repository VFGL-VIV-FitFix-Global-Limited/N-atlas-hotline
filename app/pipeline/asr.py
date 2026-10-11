from __future__ import annotations

from pathlib import Path
from typing import Protocol

import httpx
import structlog
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.config import Settings
from app.pipeline.natlas_lang import normalise_natlas_language

logger = structlog.get_logger(__name__)


class ASRError(Exception):
    """Raised when ASR transcription fails."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class ASRService(Protocol):
    async def transcribe(self, audio_path: str, language: str) -> str: ...


def _retry_transient(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500:
        return True
    return False


class NatlasASR:
    """N-ATLAS ASR via POST /v1/audio/transcriptions (OpenAI-compatible)."""

    def __init__(self, client: httpx.AsyncClient, config: Settings) -> None:
        self._client = client
        self._config = config

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception(_retry_transient),
        before_sleep=before_sleep_log(logger, "WARNING"),
    )
    async def transcribe(self, audio_path: str, language: str) -> str:
        filename = Path(audio_path).name
        suffix = Path(audio_path).suffix.lower()
        mime = {
            ".wav": "audio/wav",
            ".ogg": "audio/ogg",
            ".mp3": "audio/mpeg",
            ".m4a": "audio/mp4",
            ".webm": "audio/webm",
            ".flac": "audio/flac",
        }.get(suffix, "application/octet-stream")

        try:
            try:
                lang = normalise_natlas_language(language)
            except ValueError as e:
                raise ASRError(str(e), status_code=400) from e
            with open(audio_path, "rb") as f:
                response = await self._client.post(
                    self._config.natlas_asr_url,
                    files={"file": (filename, f, mime)},
                    data={"language": lang, "response_format": "json"},
                    headers={"Authorization": f"Bearer {self._config.natlas_api_key}"},
                    timeout=self._config.asr_timeout_seconds,
                )

            request_id = response.headers.get("X-Request-Id")

            if response.status_code == 401:
                logger.error("asr_auth_failed", request_id=request_id)
                raise ASRError("Missing or invalid N-ATLAS API key", status_code=401)

            if 400 <= response.status_code < 500:
                logger.error(
                    "asr_client_error",
                    status=response.status_code,
                    detail=response.text,
                    request_id=request_id,
                )
                raise ASRError(f"ASR client error: {response.text}", status_code=response.status_code)

            response.raise_for_status()
            data = response.json()
            text = data.get("text", "")

            logger.info(
                "asr_success",
                language=lang,
                text_length=len(text),
                request_id=request_id,
            )
            return text

        except (httpx.TimeoutException, httpx.HTTPStatusError):
            raise
        except ASRError:
            raise
        except Exception as e:
            logger.error("asr_unexpected_error", error=str(e))
            raise ASRError(f"ASR failed: {e}") from e
