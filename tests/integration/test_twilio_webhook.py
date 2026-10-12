"""Tests for the Twilio WhatsApp Sandbox webhook."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.channels.twilio_whatsapp import router
from app.config import Settings
from app.dependencies import get_orchestrator, get_session_store, get_settings
from app.models.schemas import Language


class MemorySessionStore:
    def __init__(self) -> None:
        self.data: dict[str, Language] = {}

    async def set_language(self, phone_hash: str, language: Language) -> None:
        self.data[phone_hash] = language

    async def get_language(self, phone_hash: str) -> Language | None:
        return self.data.get(phone_hash)


def _client(settings: Settings, store: MemorySessionStore | None = None) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    store = store or MemorySessionStore()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_session_store] = lambda: store
    app.dependency_overrides[get_orchestrator] = lambda: object()
    return TestClient(app)


def twilio_form_text() -> dict[str, str]:
    return {
        "Body": "hello",
        "From": "whatsapp:+2348012345678",
        "To": "whatsapp:+14155238886",
        "NumMedia": "0",
        "MediaUrl0": "",
        "MediaContentType0": "",
        "MessageSid": "SM1234567890abcdef",
    }


def twilio_form_voice_note() -> dict[str, str]:
    return {
        "Body": "",
        "From": "whatsapp:+2348012345678",
        "To": "whatsapp:+14155238886",
        "NumMedia": "1",
        "MediaUrl0": "https://api.twilio.com/2010-04-01/Accounts/AC123/Messages/MM456/Media/ME789",
        "MediaContentType0": "audio/ogg",
        "MessageSid": "SM1234567890voice",
    }


def test_text_hello_returns_welcome(settings: Settings) -> None:
    response = _client(settings).post("/twilio/webhook", data=twilio_form_text())
    assert response.status_code == 200
    assert "application/xml" in response.headers["content-type"]
    assert "<Message>" in response.text
    assert "N-ATLAS Hotline" in response.text


def test_language_selection(settings: Settings) -> None:
    form = twilio_form_text()
    form["Body"] = "1"
    response = _client(settings).post("/twilio/webhook", data=form)
    assert response.status_code == 200
    assert "Yorùbá" in response.text


def test_voice_note_returns_empty_twiml(settings: Settings, monkeypatch) -> None:
    async def noop(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("app.channels.twilio_whatsapp._process_voice_note", noop)
    response = _client(settings).post("/twilio/webhook", data=twilio_form_voice_note())
    assert response.status_code == 200
    assert "<Response></Response>" in response.text or "<Response/>" in response.text


def test_random_text_prompts_voice(settings: Settings) -> None:
    form = twilio_form_text()
    form["Body"] = "what is NIN"
    response = _client(settings).post("/twilio/webhook", data=form)
    assert response.status_code == 200
    assert "voice note" in response.text.lower()
