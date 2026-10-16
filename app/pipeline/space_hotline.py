"""Call the Hugging Face Space (same N-ATLaS + Soro as the reviewer UI)."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import structlog

from app.config import Settings

logger = structlog.get_logger(__name__)

RADIO = {"en": "English", "yo": "Yoruba", "ha": "Hausa", "ig": "Igbo"}


class SpaceHotlineError(Exception):
    def __init__(self, message: str, user_message: str | None = None) -> None:
        super().__init__(message)
        self.user_message = user_message or (
            "Sorry, I couldn't reach N-ATLaS on the Space. Please try again in a minute."
        )


def _audio_src(audio: object) -> str | None:
    if not audio:
        return None
    if isinstance(audio, str) and audio.strip():
        return audio
    if isinstance(audio, dict):
        for key in ("path", "name", "value"):
            val = audio.get(key)
            if isinstance(val, str) and val:
                return val
    return None


def _predict(audio_path: str, language: str, settings: Settings) -> tuple[str, str, str | None]:
    from gradio_client import Client, handle_file

    radio = RADIO.get(language, "English")
    token = settings.hf_token or None
    client = Client(settings.hf_space_id, hf_token=token, verbose=False)
    raw = client.predict(handle_file(audio_path), radio, api_name="/ask")
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        raise SpaceHotlineError(f"unexpected Space response: {type(raw)}")
    transcript = str(raw[0] or "")
    reply = str(raw[1] or "")
    src = _audio_src(raw[2] if len(raw) > 2 else None)
    saved: str | None = None
    if src and Path(src).is_file():
        dest_dir = Path(settings.audio_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{uuid.uuid4().hex}{Path(src).suffix or '.wav'}"
        shutil.copy(src, dest)
        saved = str(dest)
    return transcript, reply, saved


async def ask_space(audio_path: str, language: str, settings: Settings) -> tuple[str, str, str | None]:
    import asyncio

    logger.info("space_predict_start", space=settings.hf_space_id, language=language)
    try:
        transcript, reply, audio = await asyncio.to_thread(_predict, audio_path, language, settings)
    except SpaceHotlineError:
        raise
    except Exception as e:
        text = str(e)
        user = None
        low = text.lower()
        if "quota" in low or "zerogpu" in low:
            user = (
                "The Space GPU is out of daily quota. Try the web demo later, "
                "or wait until the quota resets."
            )
        elif "timeout" in low or "timed out" in low:
            user = "N-ATLaS is still loading on the Space. Please send the voice note again in a minute."
        logger.exception("space_predict_failed")
        raise SpaceHotlineError(text, user) from e
    logger.info(
        "space_predict_ok",
        transcript_len=len(transcript),
        reply_len=len(reply),
        has_audio=bool(audio),
    )
    return transcript, reply, audio
