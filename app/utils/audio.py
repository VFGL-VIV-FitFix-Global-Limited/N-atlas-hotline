import asyncio
import tempfile
from pathlib import Path

import structlog
from pydub import AudioSegment

logger = structlog.get_logger(__name__)

SUPPORTED_EXTENSIONS = {".ogg", ".wav", ".mp3", ".m4a", ".amr", ".webm"}


async def convert_to_wav(input_path: str) -> str:
    """Convert any supported audio format to WAV for ASR. Returns path to WAV file."""
    _validate_audio_file(input_path)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _to_wav_sync, input_path)


async def convert_to_ogg_opus(input_path: str) -> str:
    """Convert audio to OGG Opus for WhatsApp voice notes. Returns path."""
    _validate_audio_file(input_path)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _to_ogg_sync, input_path)


def _validate_audio_file(input_path: str) -> None:
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported audio format: {p.suffix}")


def _to_wav_sync(input_path: str) -> str:
    audio = AudioSegment.from_file(input_path)
    output_path = tempfile.mktemp(suffix=".wav")
    audio.export(output_path, format="wav")
    logger.info("audio_converted", source=input_path, target=output_path, format="wav")
    return output_path


def _to_ogg_sync(input_path: str) -> str:
    audio = AudioSegment.from_file(input_path)
    output_path = tempfile.mktemp(suffix=".ogg")
    audio.export(output_path, format="ogg", codec="libopus")
    logger.info("audio_converted", source=input_path, target=output_path, format="ogg_opus")
    return output_path
