from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Protocol

import structlog

logger = structlog.get_logger(__name__)


class TTSService(Protocol):
    async def synthesize(self, text: str, language: str) -> str: ...


class GoogleTTS:
    """gTTS → MP3 on disk under audio_dir so Africa's Talking / WhatsApp can fetch it."""

    LANGUAGE_MAP: dict[str, str] = {"en": "en", "yo": "yo", "ha": "ha", "ig": "ig"}
    SUPPORTED_LANGUAGES: set[str] = {"en", "yo", "ha"}

    def __init__(self, audio_dir: str = "data/audio") -> None:
        self._audio_dir = Path(audio_dir)
        self._audio_dir.mkdir(parents=True, exist_ok=True)

    async def synthesize(self, text: str, language: str) -> str:
        gtts_lang = self.LANGUAGE_MAP.get(language, "en")
        if gtts_lang not in self.SUPPORTED_LANGUAGES:
            logger.warning("tts_language_fallback", requested=language, fallback="en")
            gtts_lang = "en"

        loop = asyncio.get_event_loop()
        output_path = await loop.run_in_executor(None, self._generate, text, gtts_lang)
        logger.info("tts_success", language=gtts_lang, path=output_path)
        return output_path

    def _generate(self, text: str, language: str) -> str:
        from gtts import gTTS

        output_path = self._audio_dir / f"{uuid.uuid4().hex}.mp3"
        tts = gTTS(text=text, lang=language)
        tts.save(str(output_path))
        return str(output_path)
