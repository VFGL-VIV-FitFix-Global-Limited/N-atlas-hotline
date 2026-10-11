from __future__ import annotations

import asyncio
import tempfile
import uuid
from pathlib import Path

import httpx
import structlog
from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import PlainTextResponse

from app.config import Settings
from app.dependencies import get_orchestrator, get_session_store, get_settings
from app.middleware.phone_limiter import phone_limiter
from app.models.schemas import Channel, Language, PipelineRequest
from app.pipeline.orchestrator import PipelineOrchestrator
from app.services.session import SessionStore
from app.utils.hashing import hash_phone
from app.utils.phone import is_nigerian_number, normalize_ng_phone

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/voice", tags=["phone"])

LANGUAGE_MAP = {"1": Language.YORUBA, "2": Language.HAUSA, "3": Language.IGBO, "4": Language.ENGLISH}

RATE_LIMIT_MSG = "<Response><Say>Please wait a moment before trying again. Goodbye.</Say></Response>"

PROMPT_MESSAGES: dict[str, str] = {
    "en": "Please ask your question after the beep.",
    "yo": "Ẹ jọ̀wọ́ ẹ béèrè ìbéèrè yín lẹ́yìn ohùn náà.",
    "ha": "Da fatan za a yi tambayar ku bayan ƙarar.",
    "ig": "Biko jụọ ajụjụ gị mgbe olu ahụ gasịrì.",
}

WAIT_MESSAGES: dict[str, str] = {
    "en": "Thank you. Please wait while I find the answer.",
    "yo": "Ẹ ṣeun. Ẹ jọ̀wọ́ ẹ dúró díẹ̀.",
    "ha": "Na gode. Da fatan za a jira.",
    "ig": "Daalụ. Biko chere.",
}

FOLLOW_UP_MESSAGES: dict[str, str] = {
    "en": "Do you have another question? Press 1 for yes, press 2 to end.",
    "yo": "Ṣé ẹ ní ìbéèrè mìíràn? Ẹ tẹ 1 fún bẹ́ẹ̀ni, ẹ tẹ 2 láti parí.",
    "ha": "Kuna da wata tambaya? Danna 1 don eh, danna 2 don ƙare.",
    "ig": "Ị nwere ajụjụ ọzọ? Pịa 1 maka ee, pịa 2 ịkwụsị.",
}

GOODBYE_MESSAGES: dict[str, str] = {
    "en": "Thank you for using N-ATLAS Hotline. Goodbye!",
    "yo": "Ẹ ṣeun fún lílo N-ATLAS Hotline. Ó dàbọ̀!",
    "ha": "Na gode da amfani da N-ATLAS Hotline. Sai anjima!",
    "ig": "Daalụ maka iji N-ATLAS Hotline. Ka ọ dị!",
}


def _xml(body: str) -> PlainTextResponse:
    return PlainTextResponse(content=body, media_type="application/xml")


def _play_url(settings: Settings, audio_path: str) -> str:
    return f"{settings.app_base_url}/audio/{Path(audio_path).name}"


async def _place_callback(settings: Settings, caller: str) -> None:
    """Africa's Talking Voice.call(callFrom, callTo=[...]) — sync SDK, run off-loop."""

    def _call() -> None:
        import africastalking

        africastalking.initialize(settings.at_username, settings.at_api_key)
        africastalking.Voice.call(callFrom=settings.at_phone_number, callTo=[caller])

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _call)


@router.post("/callback")
async def voice_callback(
    sessionId: str = Form(""),
    callerNumber: str = Form(""),
    isActive: str = Form("1"),
    direction: str = Form(""),
    durationInSeconds: str = Form(""),
    settings: Settings = Depends(get_settings),
) -> PlainTextResponse:
    """AT voice callback. isActive=0 is hangup; duration 0 inbound = flash."""
    caller = normalize_ng_phone(callerNumber)
    logger.info(
        "voice_callback",
        session_id=sessionId,
        caller=caller,
        direction=direction,
        is_active=isActive,
        duration=durationInSeconds,
    )

    if isActive in {"0", "false", "False"}:
        if durationInSeconds in {"", "0"} and is_nigerian_number(caller):
            try:
                await _place_callback(settings, caller)
                logger.info("flash_detected_callback", caller=caller)
            except Exception:
                logger.exception("flash_callback_failed", caller=caller)
        return _xml("<Response/>")

    xml = (
        "<Response>"
        '<Say voice="en-US-Standard-C">'
        "Welcome to N-ATLAS Hotline. "
        "For Yoruba, press 1. For Hausa, press 2. For Igbo, press 3. For English, press 4."
        "</Say>"
        f'<GetDigits timeout="10" numDigits="1" callbackUrl="{settings.app_base_url}/voice/language">'
        '<Say voice="en-US-Standard-C">Please press a number now.</Say>'
        "</GetDigits>"
        "</Response>"
    )
    return _xml(xml)


@router.post("/language")
async def language_selection(
    sessionId: str = Form(""),
    callerNumber: str = Form(""),
    dtmfDigits: str = Form(""),
    settings: Settings = Depends(get_settings),
    session_store: SessionStore = Depends(get_session_store),
) -> PlainTextResponse:
    language = LANGUAGE_MAP.get(dtmfDigits, Language.ENGLISH)
    caller = normalize_ng_phone(callerNumber)
    phone_hash = hash_phone(caller, settings.phone_hash_salt)
    await session_store.set_language(phone_hash, language)

    lang = language.value
    prompt = PROMPT_MESSAGES.get(lang, PROMPT_MESSAGES["en"])
    xml = (
        f"<Response>"
        f"<Say>{prompt}</Say>"
        f'<Record maxLength="{settings.max_recording_seconds}" '
        f'trimSilence="true" playBeep="true" '
        f'callbackUrl="{settings.app_base_url}/voice/recording?lang={lang}">'
        f"</Record>"
        f"</Response>"
    )
    return _xml(xml)


@router.post("/recording")
async def handle_recording(
    request: Request,
    sessionId: str = Form(""),
    callerNumber: str = Form(""),
    recordingUrl: str = Form(""),
    settings: Settings = Depends(get_settings),
    orchestrator: PipelineOrchestrator = Depends(get_orchestrator),
) -> PlainTextResponse:
    lang = request.query_params.get("lang", "en")
    caller = normalize_ng_phone(callerNumber)
    phone_hash = hash_phone(caller, settings.phone_hash_salt)

    if not phone_limiter.allow(phone_hash, settings.rate_limit_calls_per_minute):
        return _xml(RATE_LIMIT_MSG)

    if not recordingUrl:
        return _xml("<Response><Say>No recording received. Goodbye.</Say></Response>")

    try:
        async with httpx.AsyncClient() as client:
            audio_resp = await client.get(recordingUrl, timeout=15)
        audio_path = tempfile.mktemp(suffix=".mp3")
        with open(audio_path, "wb") as f:
            f.write(audio_resp.content)

        result = await orchestrator.process(
            PipelineRequest(
                session_id=sessionId or str(uuid.uuid4()),
                channel=Channel.PHONE,
                audio_path=audio_path,
                language=Language(lang),
                phone_number=caller,
            )
        )

        follow_up = FOLLOW_UP_MESSAGES.get(lang, FOLLOW_UP_MESSAGES["en"])
        if result.response_audio_path:
            play = _play_url(settings, result.response_audio_path)
            xml = (
                f"<Response>"
                f'<Play url="{play}"/>'
                f'<GetDigits timeout="8" numDigits="1" '
                f'callbackUrl="{settings.app_base_url}/voice/followup?lang={lang}">'
                f"<Say>{follow_up}</Say>"
                f"</GetDigits>"
                f"</Response>"
            )
        else:
            xml = (
                f"<Response>"
                f"<Say>{result.response_text}</Say>"
                f'<GetDigits timeout="8" numDigits="1" '
                f'callbackUrl="{settings.app_base_url}/voice/followup?lang={lang}">'
                f"<Say>{follow_up}</Say>"
                f"</GetDigits>"
                f"</Response>"
            )
        return _xml(xml)
    except Exception:
        logger.exception("voice_recording_handler_failed", session_id=sessionId)
        return _xml("<Response><Say>I'm sorry, something went wrong. Please try again.</Say></Response>")


@router.post("/followup")
async def handle_followup(
    request: Request,
    dtmfDigits: str = Form(""),
    settings: Settings = Depends(get_settings),
) -> PlainTextResponse:
    lang = request.query_params.get("lang", "en")
    if dtmfDigits == "1":
        prompt = PROMPT_MESSAGES.get(lang, PROMPT_MESSAGES["en"])
        xml = (
            f"<Response>"
            f"<Say>{prompt}</Say>"
            f'<Record maxLength="{settings.max_recording_seconds}" '
            f'trimSilence="true" playBeep="true" '
            f'callbackUrl="{settings.app_base_url}/voice/recording?lang={lang}">'
            f"</Record>"
            f"</Response>"
        )
        return _xml(xml)
    return _xml(f"<Response><Say>{GOODBYE_MESSAGES.get(lang, GOODBYE_MESSAGES['en'])}</Say></Response>")


@router.post("/flash")
async def handle_flash(
    callerNumber: str = Form(""),
    settings: Settings = Depends(get_settings),
) -> PlainTextResponse:
    caller = normalize_ng_phone(callerNumber)
    if not is_nigerian_number(caller):
        logger.warning("flash_invalid_number", caller=caller)
        return _xml("<Response><Reject/></Response>")

    logger.info("flash_callback_initiated", caller=caller)
    try:
        await _place_callback(settings, caller)
        logger.info("flash_outbound_call_placed", caller=caller)
    except Exception:
        logger.exception("flash_callback_failed", caller=caller)
    return _xml("<Response/>")
