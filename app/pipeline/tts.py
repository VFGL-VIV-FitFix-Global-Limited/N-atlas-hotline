from __future__ import annotations

import asyncio
import re
import uuid
import wave
from pathlib import Path
from typing import Protocol

import structlog

logger = structlog.get_logger(__name__)

# Official N-ATLaS has no TTS. Same Soro MMS/VITS checkpoints as the Space.
SORO_IDS = {
    "yo": "Shinzmann/soro-tts-yor",
    "ha": "Shinzmann/soro-tts-hau",
    "ig": "Shinzmann/soro-tts-ibo",
}


class TTSService(Protocol):
    async def synthesize(self, text: str, language: str) -> str: ...


class GoogleTTS:
    """gTTS → MP3 on disk (English, and fallback if Soro fails)."""

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
        logger.info("tts_success", language=gtts_lang, path=output_path, engine="gtts")
        return output_path

    def _generate(self, text: str, language: str) -> str:
        from gtts import gTTS

        output_path = self._audio_dir / f"{uuid.uuid4().hex}.mp3"
        tts = gTTS(text=text, lang=language)
        tts.save(str(output_path))
        return str(output_path)


class HotlineTTS:
    """Yoruba / Igbo / Hausa → Soro-TTS on CPU; English → gTTS. Not N-ATLaS."""

    def __init__(self, audio_dir: str = "data/audio") -> None:
        self._audio_dir = Path(audio_dir)
        self._audio_dir.mkdir(parents=True, exist_ok=True)
        self._gtts = GoogleTTS(audio_dir)
        self._soro: dict = {}

    async def synthesize(self, text: str, language: str) -> str:
        text = (text or "").strip()
        if not text:
            return await self._gtts.synthesize(" ", "en")
        lang = (language or "en").split("-")[0].lower()
        if lang not in SORO_IDS:
            return await self._gtts.synthesize(text, "en")
        loop = asyncio.get_event_loop()
        try:
            path = await loop.run_in_executor(None, self._soro_generate, text, lang)
            logger.info("tts_success", language=lang, path=path, engine="soro")
            return path
        except Exception:
            logger.exception("soro_tts_failed_falling_back_gtts", language=lang)
            return await self._gtts.synthesize(text, lang)

    def _ensure_soro(self, lang: str) -> tuple:
        if lang not in self._soro:
            import torch
            from transformers import AutoTokenizer, VitsModel

            model = VitsModel.from_pretrained(SORO_IDS[lang]).to("cpu").eval()
            tokenizer = AutoTokenizer.from_pretrained(SORO_IDS[lang])
            self._soro[lang] = (model, tokenizer)
            logger.info("soro_loaded", language=lang, repo=SORO_IDS[lang])
        return self._soro[lang]

    def _soro_generate(self, text: str, lang: str) -> str:
        import numpy as np
        import torch

        model, tokenizer = self._ensure_soro(lang)
        chunks = [c.strip() for c in re.split(r"(?<=[.!?])\s+", text) if c.strip()]
        if not chunks:
            chunks = [text]
        waves: list = []
        with torch.no_grad():
            for chunk in chunks:
                inputs = tokenizer(chunk, return_tensors="pt")
                waves.append(model(**inputs).waveform[0].cpu().numpy())
        audio = np.concatenate(waves)
        output_path = self._audio_dir / f"{uuid.uuid4().hex}.wav"
        pcm = np.clip(np.asarray(audio).squeeze() * 32767.0, -32768, 32767).astype(np.int16)
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(model.config.sampling_rate))
            wf.writeframes(pcm.tobytes())
        return str(output_path)
