from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.channels import whatsapp
from app.channels.whatsapp import router
from app.config import Settings
from app.dependencies import get_orchestrator, get_session_store, get_settings
from app.models.schemas import Language
from app.services.session import SessionStore


class MemorySessionStore:
    def __init__(self) -> None:
        self.data: dict[str, Language] = {}

    async def set_language(self, phone_hash: str, language: Language) -> None:
        self.data[phone_hash] = language

    async def get_language(self, phone_hash: str) -> Language | None:
        return self.data.get(phone_hash)


def _app(settings: Settings, store: MemorySessionStore | None = None) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    store = store or MemorySessionStore()

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_session_store] = lambda: store
    app.dependency_overrides[get_orchestrator] = lambda: object()
    return TestClient(app)


def test_webhook_verify_ok(settings: Settings) -> None:
    client = _app(settings)
    resp = client.get(
        "/whatsapp/webhook",
        params={"hub.mode": "subscribe", "hub.verify_token": "verify-me", "hub.challenge": "challenge123"},
    )
    assert resp.status_code == 200
    assert resp.text == "challenge123"


def test_webhook_verify_bad_token(settings: Settings) -> None:
    client = _app(settings)
    resp = client.get(
        "/whatsapp/webhook",
        params={"hub.mode": "subscribe", "hub.verify_token": "wrong", "hub.challenge": "x"},
    )
    assert resp.status_code == 403


def test_incoming_voice_note_returns_200(settings: Settings, monkeypatch) -> None:
    store = MemorySessionStore()

    async def fake_get(phone_hash: str) -> Language:
        return Language.ENGLISH

    store.get_language = fake_get  # type: ignore[method-assign]

    sent: list[str] = []

    async def fake_text(to: str, text: str, settings: Settings) -> None:
        sent.append(text)

    monkeypatch.setattr(whatsapp, "send_whatsapp_text", fake_text)
    client = _app(settings, store)
    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {"from": "2348012345678", "type": "audio", "audio": {"id": "media1"}}
                            ]
                        }
                    }
                ]
            }
        ]
    }
    resp = client.post("/whatsapp/webhook", json=payload)
    assert resp.status_code == 200
    assert any("Processing" in t for t in sent)


def test_incoming_text_asks_for_voice_note(settings: Settings, monkeypatch) -> None:
    store = MemorySessionStore()

    async def fake_get(phone_hash: str) -> Language:
        return Language.ENGLISH

    store.get_language = fake_get  # type: ignore[method-assign]
    sent: list[str] = []

    async def fake_text(to: str, text: str, settings: Settings) -> None:
        sent.append(text)

    monkeypatch.setattr(whatsapp, "send_whatsapp_text", fake_text)
    client = _app(settings, store)
    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {"from": "2348012345678", "type": "text", "text": {"body": "hello"}}
                            ]
                        }
                    }
                ]
            }
        ]
    }
    resp = client.post("/whatsapp/webhook", json=payload)
    assert resp.status_code == 200
    assert any("voice note" in t.lower() for t in sent)


def test_language_button_saves_preference(settings: Settings, monkeypatch) -> None:
    store = MemorySessionStore()
    sent: list[str] = []

    async def fake_text(to: str, text: str, settings: Settings) -> None:
        sent.append(text)

    monkeypatch.setattr(whatsapp, "send_whatsapp_text", fake_text)
    client = _app(settings, store)
    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {
                                    "from": "2348012345678",
                                    "type": "interactive",
                                    "interactive": {"button_reply": {"id": "yo"}},
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }
    resp = client.post("/whatsapp/webhook", json=payload)
    assert resp.status_code == 200
    assert sent
    assert Language.YORUBA in store.data.values()
