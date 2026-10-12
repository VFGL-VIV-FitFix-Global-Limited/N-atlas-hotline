"""
Twilio WhatsApp Sandbox webhook handler.

Key differences from Meta WhatsApp Cloud API (whatsapp.py):
- Twilio sends POST data as application/x-www-form-urlencoded (form fields), NOT JSON
- Voice notes arrive as: NumMedia="1", MediaUrl0="https://...", MediaContentType0="audio/ogg"
- Sender format: "whatsapp:+2348012345678" (strip "whatsapp:" prefix to get phone number)
- Replies use TwiML XML response (not a separate API call)
- Webhook validation uses Twilio request signature (X-Twilio-Signature header)
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import tempfile
import uuid
from base64 import b64encode
from html import escape
from pathlib import Path

import httpx
import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, Form, Request
from fastapi.responses import Response as FastAPIResponse

from app.config import Settings
from app.dependencies import get_orchestrator, get_session_store, get_settings
from app.models.schemas import Channel, Language, PipelineRequest
from app.pipeline.orchestrator import PipelineOrchestrator
from app.services.session import SessionStore
from app.utils.hashing import hash_phone
from app.utils.phone import normalize_ng_phone

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/twilio", tags=["twilio-whatsapp"])

LANGUAGE_MAP = {
    "yoruba": "yo",
    "hausa": "ha",
    "igbo": "ig",
    "english": "en",
    "1": "yo",
    "2": "ha",
    "3": "ig",
    "4": "en",
}

LANG_NAMES = {"yo": "Yorùbá", "ha": "Hausa", "ig": "Igbo", "en": "English"}


def validate_twilio_signature(
    url: str,
    params: dict,
    signature: str,
    auth_token: str,
) -> bool:
    """Validate incoming Twilio webhook request."""
    if not signature or not auth_token:
        return False

    s = url
    for key in sorted(params.keys()):
        s += key
        value = params[key]
        if isinstance(value, list):
            for item in value:
                s += item
        else:
            s += value

    mac = hmac.new(auth_token.strip().encode("utf-8"), s.encode("utf-8"), hashlib.sha1)
    computed = b64encode(mac.digest()).decode("utf-8")
    try:
        return hmac.compare_digest(computed, signature)
    except Exception:
        return False


def twilio_signed_urls(request: Request, settings: Settings) -> list[str]:
    """URLs Twilio may have signed. Railway's internal request.url is usually http."""
    path = request.url.path
    query = request.url.query
    suffix = f"{path}?{query}" if query else path

    candidates: list[str] = []
    if settings.app_base_url:
        candidates.append(f"{settings.app_base_url.rstrip('/')}{suffix}")

    proto = (request.headers.get("x-forwarded-proto") or request.url.scheme).split(",")[0].strip()
    host = (
        request.headers.get("x-forwarded-host")
        or request.headers.get("host")
        or request.url.netloc
    ).split(",")[0].strip()
    if proto and host:
        candidates.append(f"{proto}://{host}{suffix}")

    candidates.append(str(request.url).split("#")[0])

    seen: list[str] = []
    for url in candidates:
        url = url.rstrip("/") if url.endswith("/") and "?" not in url else url
        if url and url not in seen:
            seen.append(url)
        if url.startswith("http://"):
            https = "https://" + url[len("http://") :]
            if https not in seen:
                seen.append(https)
    return seen


def signature_is_valid(request: Request, params: dict, settings: Settings) -> bool:
    signature = request.headers.get("X-Twilio-Signature", "")
    token = settings.twilio_auth_token
    for url in twilio_signed_urls(request, settings):
        if validate_twilio_signature(url, params, signature, token):
            return True
    return False


def twiml_reply(message: str) -> FastAPIResponse:
    """Return a TwiML XML response that sends a WhatsApp text reply."""
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"<Message>{escape(message)}</Message>"
        "</Response>"
    )
    return FastAPIResponse(content=xml, media_type="application/xml")


def twiml_empty() -> FastAPIResponse:
    """Return an empty TwiML response (no reply, just acknowledge)."""
    xml = '<?xml version="1.0" encoding="UTF-8"?><Response></Response>'
    return FastAPIResponse(content=xml, media_type="application/xml")


def extract_phone_number(from_field: str) -> str:
    """Extract phone number from Twilio's whatsapp:+E.164 format."""
    if from_field.startswith("whatsapp:"):
        return from_field[len("whatsapp:") :]
    return from_field


def _form_params_for_signature(form_data: object) -> dict[str, str]:
    params: dict[str, str] = {}
    for key, value in form_data.items():
        params[key] = value if isinstance(value, str) else str(value)
    return params


@router.post("/webhook")
async def twilio_whatsapp_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    orchestrator: PipelineOrchestrator = Depends(get_orchestrator),
    session_store: SessionStore = Depends(get_session_store),
    settings: Settings = Depends(get_settings),
    Body: str = Form(default=""),
    From: str = Form(default=""),
    To: str = Form(default=""),
    NumMedia: str = Form(default="0"),
    MediaUrl0: str = Form(default=""),
    MediaContentType0: str = Form(default=""),
    MessageSid: str = Form(default=""),
) -> FastAPIResponse:
    """Handle incoming WhatsApp messages via Twilio Sandbox (form-data)."""
    if settings.twilio_auth_token:
        form_data = await request.form()
        params = _form_params_for_signature(form_data)
        if not signature_is_valid(request, params, settings):
            logger.warning(
                "twilio_signature_invalid",
                message_sid=MessageSid,
                signed_urls=twilio_signed_urls(request, settings),
            )
            return twiml_empty()

    phone_number = normalize_ng_phone(extract_phone_number(From))
    phone_hash = hash_phone(phone_number, settings.phone_hash_salt)

    logger.info(
        "twilio_message_received",
        phone=phone_number[-4:] if len(phone_number) >= 4 else phone_number,
        has_media=NumMedia != "0",
        media_type=MediaContentType0 if NumMedia != "0" else None,
        body_length=len(Body),
        message_sid=MessageSid,
    )

    if NumMedia != "0" and MediaContentType0.startswith("audio/"):
        background_tasks.add_task(
            _process_voice_note,
            phone_number=phone_number,
            phone_hash=phone_hash,
            media_url=MediaUrl0,
            media_content_type=MediaContentType0,
            message_sid=MessageSid,
            orchestrator=orchestrator,
            session_store=session_store,
            settings=settings,
        )
        return twiml_empty()

    body_lower = Body.strip().lower()

    if body_lower in LANGUAGE_MAP:
        lang_code = LANGUAGE_MAP[body_lower]
        await session_store.set_language(phone_hash, Language(lang_code))
        return twiml_reply(
            f"✅ Language set to {LANG_NAMES[lang_code]}.\n\n"
            "Send a *voice note* with your question and I'll respond!"
        )

    if body_lower in ("hi", "hello", "help", "start", "menu"):
        return twiml_reply(
            "👋 Welcome to *N-ATLAS Hotline*!\n\n"
            "I answer questions about government services, health, agriculture "
            "and market prices in your language.\n\n"
            "🗣️ *Send a voice note* with your question!\n\n"
            "Or choose your language:\n"
            "1️⃣ Yorùbá\n"
            "2️⃣ Hausa\n"
            "3️⃣ Igbo\n"
            "4️⃣ English"
        )

    return twiml_reply(
        "🎤 Please send a *voice note* with your question.\n\n"
        "I work best with voice! You can also type a number to set your language:\n"
        "1=Yorùbá  2=Hausa  3=Igbo  4=English"
    )


async def _process_voice_note(
    phone_number: str,
    phone_hash: str,
    media_url: str,
    media_content_type: str,
    message_sid: str,
    orchestrator: PipelineOrchestrator,
    session_store: SessionStore,
    settings: Settings,
) -> None:
    """Download Twilio media, run the pipeline, reply via REST API."""
    log = logger.bind(
        message_sid=message_sid,
        phone=phone_number[-4:] if len(phone_number) >= 4 else phone_number,
    )
    tmp_audio_path: str | None = None

    try:
        stored = await session_store.get_language(phone_hash)
        language = stored or Language.ENGLISH

        log.info("downloading_voice_note", media_type=media_content_type)

        response: httpx.Response | None = None
        async with httpx.AsyncClient(timeout=30.0) as client:
            last_error: Exception | None = None
            for attempt in range(5):
                try:
                    response = await client.get(
                        media_url,
                        auth=(settings.twilio_account_sid, settings.twilio_auth_token),
                        follow_redirects=True,
                    )
                    if response.status_code in {404, 409, 425}:
                        log.warning(
                            "twilio_media_not_ready",
                            status=response.status_code,
                            attempt=attempt + 1,
                        )
                        await asyncio.sleep(1.5)
                        continue
                    response.raise_for_status()
                    last_error = None
                    break
                except httpx.HTTPStatusError as e:
                    last_error = e
                    if e.response.status_code >= 500:
                        await asyncio.sleep(1.5)
                        continue
                    raise
            if response is None or last_error is not None or response.status_code >= 400:
                if last_error:
                    raise last_error
                response.raise_for_status()  # type: ignore[union-attr]

        ext_map = {
            "audio/ogg": ".ogg",
            "audio/mpeg": ".mp3",
            "audio/amr": ".amr",
            "audio/wav": ".wav",
            "audio/x-wav": ".wav",
        }
        ext = ext_map.get(media_content_type, ".ogg")

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(response.content)
            tmp_audio_path = tmp.name

        log.info("voice_note_downloaded", path=tmp_audio_path, size=len(response.content))

        result = await orchestrator.process(
            PipelineRequest(
                session_id=str(uuid.uuid4()),
                channel=Channel.WHATSAPP,
                audio_path=tmp_audio_path,
                language=language,
                phone_number=phone_number,
            )
        )
        tmp_audio_path = None

        # Text only — sandbox 429s if we also send the TTS audio as a second message.
        await _send_twilio_message(
            to=f"whatsapp:{phone_number}",
            body=result.response_text,
            settings=settings,
        )

        log.info(
            "voice_note_processed",
            language=language.value,
            intent=result.intent_domain.value,
            response_length=len(result.response_text),
        )

    except Exception as e:
        log.error("voice_note_processing_failed", error=str(e))
        try:
            await _send_twilio_message(
                to=f"whatsapp:{phone_number}",
                body="⚠️ Sorry, I couldn't process your voice note. Please try again.",
                settings=settings,
            )
        except Exception:
            log.error("error_reply_failed")

    finally:
        if tmp_audio_path:
            try:
                os.unlink(tmp_audio_path)
            except OSError:
                pass


async def _send_twilio_message(
    to: str,
    body: str,
    settings: Settings,
    media_url: str | None = None,
) -> None:
    """Send a WhatsApp message via Twilio REST API (async replies)."""
    url = f"https://api.twilio.com/2010-04-01/Accounts/{settings.twilio_account_sid}/Messages.json"

    data = {
        "To": to,
        "From": f"whatsapp:{settings.twilio_whatsapp_number}",
        "Body": body,
    }
    if media_url:
        data["MediaUrl"] = media_url

    async with httpx.AsyncClient(timeout=15.0) as client:
        last_error: Exception | None = None
        for attempt in range(6):
            response = await client.post(
                url,
                data=data,
                auth=(settings.twilio_account_sid, settings.twilio_auth_token),
            )
            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After") or 20)
                logger.warning("twilio_rate_limited", attempt=attempt + 1, retry_after=retry_after)
                await asyncio.sleep(min(retry_after, 45))
                last_error = httpx.HTTPStatusError(
                    "429 Too Many Requests", request=response.request, response=response
                )
                continue
            response.raise_for_status()
            last_error = None
            break
        if last_error:
            raise last_error

    suffix = to[-4:] if len(to) >= 4 else to
    logger.info("twilio_message_sent", to=suffix, body_length=len(body))
