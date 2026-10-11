from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.channels.phone import router
from app.config import Settings
from app.dependencies import get_orchestrator, get_session_store, get_settings
from app.models.schemas import Language


class MemorySessionStore:
    def __init__(self) -> None:
        self.saved: dict[str, Language] = {}

    async def set_language(self, phone_hash: str, language: Language) -> None:
        self.saved[phone_hash] = language

    async def get_language(self, phone_hash: str) -> Language | None:
        return self.saved.get(phone_hash)


def _client(settings: Settings, store: MemorySessionStore | None = None) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    store = store or MemorySessionStore()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_session_store] = lambda: store
    app.dependency_overrides[get_orchestrator] = lambda: object()
    return TestClient(app)


def test_incoming_call_returns_ivr(settings: Settings) -> None:
    client = _client(settings)
    resp = client.post("/voice/callback", data={"sessionId": "abc", "callerNumber": "+2348012345678"})
    assert resp.status_code == 200
    assert "Yoruba" in resp.text
    assert "GetDigits" in resp.text


def test_dtmf_language_selection(settings: Settings) -> None:
    store = MemorySessionStore()
    client = _client(settings, store)
    resp = client.post(
        "/voice/language",
        data={"sessionId": "abc", "callerNumber": "+2348012345678", "dtmfDigits": "1"},
    )
    assert resp.status_code == 200
    assert "Record" in resp.text
    assert Language.YORUBA in store.saved.values()


def test_flash_rejects_non_nigerian(settings: Settings) -> None:
    client = _client(settings)
    resp = client.post("/voice/flash", data={"callerNumber": "+15555550100"})
    assert resp.status_code == 200
    assert "Reject" in resp.text


def test_flash_initiates_outbound(settings: Settings, monkeypatch) -> None:
    called: list[tuple[str, str]] = []

    class FakeVoice:
        @staticmethod
        def call(callFrom: str, callTo: list[str]) -> None:
            called.append((callFrom, tuple(callTo)))

    class FakeAT:
        Voice = FakeVoice

        @staticmethod
        def initialize(username: str, api_key: str) -> None:
            return None

    import sys
    import types

    fake_mod = types.ModuleType("africastalking")
    fake_mod.initialize = FakeAT.initialize  # type: ignore[attr-defined]
    fake_mod.Voice = FakeVoice  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "africastalking", fake_mod)

    client = _client(settings)
    resp = client.post("/voice/flash", data={"callerNumber": "+2348012345678"})
    assert resp.status_code == 200
    assert called == [("+2348000000000", ("+2348012345678",))]
