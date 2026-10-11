from __future__ import annotations

import hashlib
import hmac
import json
import tempfile
import uuid
from pathlib import Path

import httpx
import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from app.config import Settings
from app.dependencies import get_orchestrator, get_session_store, get_settings
from app.models.schemas import Channel, Language, PipelineRequest
from app.pipeline.orchestrator import PipelineOrchestrator
from app.services.session import SessionStore
from app.middleware.phone_limiter import phone_limiter
from app.utils.audio import convert_to_ogg_opus
from app.utils.hashing import hash_phone
from app.utils.phone import normalize_ng_phone

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/whatsapp", tags=["whatsapp"])

LANGUAGE_BUTTONS = {
    "yo": "Yoruba",
    "ha": "Hausa",
    "ig": "Igbo",
    "en": "English",
}

WELCOME_TEXT = (
    "Welcome to N-ATLAS Hotline! 🇳🇬\n\n"
    "I can help you with government services, health, market prices and more.\n\n"
    "Choose your language:"
)

VOICE_PROMPT: dict[str, str] = {
    "en": "Send me a voice note with your question. I can help with government services, health, market prices and more.",
    "yo": "Ẹ fi ohùn yín ránṣẹ́ pẹ̀lú ìbéèrè yín. Mo lè ran yín lọ́wọ́ pẹ̀lú iṣẹ́ ìjọba, ìlera, owó ọjà àti púpọ̀ síi.",
    "ha": "Aiko mana saƙon murya tare da tambayar ku. Zan iya taimaka muku da sabis na gwamnati, lafiya, farashin kasuwa da ƙari.",
    "ig": "Zitere m ozi olu nwere ajụjụ gị. Enwere m ike inyere gị aka na ọrụ gọọmentị, ahụike, ọnụahịa ahịa na ndị ọzọ.",
}

SEND_VOICE_NOTE: dict[str, str] = {
    "en": "Please send a voice note — I respond to spoken questions, not text.",
    "yo": "Ẹ jọ̀wọ́ ẹ fi ohùn ránṣẹ́ — mo máa ń dáhùn ìbéèrè ohùn, kìí ṣe ọ̀rọ̀ títẹ̀.",
    "ha": "Da fatan za a aiko saƙon murya — ina amsa tambayoyin da aka yi da baki, ba rubutu ba.",
    "ig": "Biko zitere ozi olu — ana m aza ajụjụ a kwuru okwu, ọ bụghị ederede.",
}

PROCESSING_MSG: dict[str, str] = {
    "en": "⏳ Processing your question...",
    "yo": "⏳ Mo ń ṣe àyẹ̀wò ìbéèrè yín...",
    "ha": "⏳ Ina aiwatar da tambayar ku...",
    "ig": "⏳ Ana m ahazi ajụjụ gị...",
}


def _wa_headers(settings: Settings) -> dict[str, str]:
    return {"Authorization": f"Bearer {settings.wa_access_token}"}


def verify_whatsapp_signature(raw_body: bytes, header: str | None, app_secret: str) -> bool:
    """HMAC-SHA256 of raw POST body with Meta App Secret. Skip if secret unset (local)."""
    if not app_secret:
        return True
    if not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(app_secret.encode(), raw_body, hashlib.sha256).hexdigest()
    received = header.split("=", 1)[1]
    return hmac.compare_digest(expected, received)


async def send_whatsapp_text(to: str, text: str, settings: Settings) -> None:
    async with httpx.AsyncClient() as client:
        await client.post(
            f"{settings.wa_graph_base}/{settings.wa_phone_number_id}/messages",
            json={
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": to,
                "type": "text",
                "text": {"body": text},
            },
            headers=_wa_headers(settings),
            timeout=10,
        )


async def send_language_list(to: str, settings: Settings) -> None:
    """Reply buttons max out at 3; 4 languages need an interactive list."""
    async with httpx.AsyncClient() as client:
        await client.post(
            f"{settings.wa_graph_base}/{settings.wa_phone_number_id}/messages",
            json={
                "messaging_product": "whatsapp",
                "to": to,
                "type": "interactive",
                "interactive": {
                    "type": "list",
                    "body": {"text": WELCOME_TEXT},
                    "action": {
                        "button": "Choose language",
                        "sections": [
                            {
                                "title": "Language",
                                "rows": [
                                    {"id": code, "title": name}
                                    for code, name in LANGUAGE_BUTTONS.items()
                                ],
                            }
                        ],
                    },
                },
            },
            headers=_wa_headers(settings),
            timeout=10,
        )


async def send_whatsapp_audio(to: str, audio_path: str, settings: Settings) -> None:
    """Upload OGG/Opus (voice-note compatible) or fall back to a public /audio link."""
    try:
        ogg_path = await convert_to_ogg_opus(audio_path)
        async with httpx.AsyncClient() as client:
            with open(ogg_path, "rb") as f:
                upload_resp = await client.post(
                    f"{settings.wa_graph_base}/{settings.wa_phone_number_id}/media",
                    files={"file": ("response.ogg", f, "audio/ogg")},
                    data={"messaging_product": "whatsapp", "type": "audio/ogg"},
                    headers=_wa_headers(settings),
                    timeout=15,
                )
            media_id = upload_resp.json().get("id")
            if not media_id:
                raise ValueError(upload_resp.text)
            await client.post(
                f"{settings.wa_graph_base}/{settings.wa_phone_number_id}/messages",
                json={
                    "messaging_product": "whatsapp",
                    "to": to,
                    "type": "audio",
                    "audio": {"id": media_id},
                },
                headers=_wa_headers(settings),
                timeout=10,
            )
            return
    except Exception:
        logger.warning("whatsapp_audio_upload_failed_using_link", path=audio_path)

    link = f"{settings.app_base_url}/audio/{Path(audio_path).name}"
    async with httpx.AsyncClient() as client:
        await client.post(
            f"{settings.wa_graph_base}/{settings.wa_phone_number_id}/messages",
            json={
                "messaging_product": "whatsapp",
                "to": to,
                "type": "audio",
                "audio": {"link": link},
            },
            headers=_wa_headers(settings),
            timeout=10,
        )


async def download_whatsapp_media(media_id: str, settings: Settings) -> str:
    async with httpx.AsyncClient() as client:
        media_resp = await client.get(
            f"{settings.wa_graph_base}/{media_id}",
            params={"phone_number_id": settings.wa_phone_number_id},
            headers=_wa_headers(settings),
            timeout=10,
        )
        media_resp.raise_for_status()
        media_url = media_resp.json()["url"]
        audio_resp = await client.get(media_url, headers=_wa_headers(settings), timeout=15)
        audio_resp.raise_for_status()

    audio_path = tempfile.mktemp(suffix=".ogg")
    with open(audio_path, "wb") as f:
        f.write(audio_resp.content)
    return audio_path


# --- Webhook endpoints ---

@router.get("/webhook", response_model=None)
async def verify_webhook(
    hub_mode: str = Query("", alias="hub.mode"),
    hub_verify_token: str = Query("", alias="hub.verify_token"),
    hub_challenge: str = Query("", alias="hub.challenge"),
    settings: Settings = Depends(get_settings),
) -> PlainTextResponse | JSONResponse:
    """WhatsApp webhook verification."""
    if hub_mode == "subscribe" and hub_verify_token == settings.wa_verify_token:
        logger.info("whatsapp_webhook_verified")
        return PlainTextResponse(content=hub_challenge)
    logger.warning("whatsapp_webhook_verification_failed")
    return JSONResponse(status_code=403, content={"error": "Verification failed"})


@router.post("/webhook")
async def handle_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
    orchestrator: PipelineOrchestrator = Depends(get_orchestrator),
    session_store: SessionStore = Depends(get_session_store),
) -> JSONResponse:
    """Handle incoming WhatsApp messages."""
    raw = await request.body()
    if not verify_whatsapp_signature(
        raw, request.headers.get("x-hub-signature-256"), settings.wa_app_secret
    ):
        logger.warning("whatsapp_bad_signature")
        return JSONResponse(status_code=403, content={"error": "Invalid signature"})

    body = json.loads(raw or b"{}")

    try:
        entry = body.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})
        messages = value.get("messages", [])
    except (IndexError, KeyError):
        return JSONResponse(content={"status": "ok"})

    if not messages:
        return JSONResponse(content={"status": "ok"})

    message = messages[0]
    from_number = normalize_ng_phone(message.get("from", ""))
    msg_type = message.get("type", "")
    phone_hash = hash_phone(from_number, settings.phone_hash_salt)

    if not phone_limiter.allow(phone_hash, settings.rate_limit_whatsapp_per_minute):
        await send_whatsapp_text(from_number, "Please wait a moment before trying again.", settings)
        return JSONResponse(content={"status": "ok"})

    if msg_type == "interactive":
        interactive = message.get("interactive", {})
        reply_id = (interactive.get("list_reply") or interactive.get("button_reply") or {}).get("id", "")
        if reply_id in LANGUAGE_BUTTONS:
            language = Language(reply_id)
            await session_store.set_language(phone_hash, language)
            await send_whatsapp_text(from_number, VOICE_PROMPT.get(reply_id, VOICE_PROMPT["en"]), settings)
            return JSONResponse(content={"status": "ok"})

    language = await session_store.get_language(phone_hash)
    if not language:
        await send_language_list(from_number, settings)
        return JSONResponse(content={"status": "ok"})

    lang = language.value

    # Handle voice notes
    if msg_type == "audio":
        media_id = message.get("audio", {}).get("id", "")
        if media_id:
            # Send processing indicator, then handle in background
            await send_whatsapp_text(from_number, PROCESSING_MSG.get(lang, PROCESSING_MSG["en"]), settings)
            background_tasks.add_task(
                _process_voice_note,
                media_id=media_id,
                from_number=from_number,
                language=language,
                settings=settings,
                orchestrator=orchestrator,
            )
        return JSONResponse(content={"status": "ok"})

    # Handle text messages — ask for voice note
    if msg_type == "text":
        text_body = message.get("text", {}).get("body", "").strip().lower()

        # Allow text-based language change
        lang_map = {"yoruba": "yo", "hausa": "ha", "igbo": "ig", "english": "en"}
        for name, code in lang_map.items():
            if text_body == name:
                new_lang = Language(code)
                await session_store.set_language(phone_hash, new_lang)
                prompt = VOICE_PROMPT.get(code, VOICE_PROMPT["en"])
                await send_whatsapp_text(from_number, prompt, settings)
                return JSONResponse(content={"status": "ok"})

        await send_whatsapp_text(
            from_number,
            SEND_VOICE_NOTE.get(lang, SEND_VOICE_NOTE["en"]),
            settings,
        )
        return JSONResponse(content={"status": "ok"})

    return JSONResponse(content={"status": "ok"})


async def _process_voice_note(
    media_id: str,
    from_number: str,
    language: Language,
    settings: Settings,
    orchestrator: PipelineOrchestrator,
) -> None:
    """Background task: download voice note, process, send response."""
    try:
        audio_path = await download_whatsapp_media(media_id, settings)

        pipeline_request = PipelineRequest(
            session_id=str(uuid.uuid4()),
            channel=Channel.WHATSAPP,
            audio_path=audio_path,
            language=language,
            phone_number=from_number,
        )

        result = await orchestrator.process(pipeline_request)

        # Send voice note response
        if result.response_audio_path:
            await send_whatsapp_audio(from_number, result.response_audio_path, settings)

        # Also send short text (for those who can read)
        await send_whatsapp_text(from_number, result.response_text, settings)

    except Exception:
        logger.exception("whatsapp_voice_processing_failed", from_number=from_number)
        lang = language.value
        error_msg = {
            "en": "Sorry, I couldn't process your voice note. Please try again.",
            "yo": "Ẹ má bínú, mi ò lè gbọ́ ohùn yín. Ẹ gbìyànjú lẹ́ẹ̀kan si.",
            "ha": "Yi hakuri, ban iya sarrafa saƙon murya ba. Da fatan za a sake gwadawa.",
            "ig": "Ndo, enweghị m ike ịhazi ozi olu gị. Biko nwaa ọzọ.",
        }
        await send_whatsapp_text(from_number, error_msg.get(lang, error_msg["en"]), settings)
