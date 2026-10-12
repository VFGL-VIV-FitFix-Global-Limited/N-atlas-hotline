from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio

from app.config import Environment, Settings
from app.models.database import Database
from app.pipeline.intent import IntentClassifier


@pytest.fixture
def settings() -> Settings:
    return Settings(
        natlas_backend="gateway",
        natlas_asr_url="https://api.natlas.ng/asr",
        natlas_llm_url="https://api.natlas.ng/llm",
        natlas_api_key="test-key",
        hf_token="",
        at_username="test",
        at_api_key="test",
        at_phone_number="+2348000000000",
        wa_phone_number_id="123",
        wa_access_token="test-token",
        wa_verify_token="verify-me",
        wa_app_secret="",
        twilio_account_sid="",
        twilio_auth_token="",
        twilio_whatsapp_number="+14155238886",
        app_base_url="https://example.com",
        database_path=":memory:",
        knowledge_path="tests/fixtures/sample_knowledge/",
        faiss_index_path="data/test_faiss",
        log_level="INFO",
        environment=Environment.DEVELOPMENT,
        api_secret_key="test-secret-key",
        phone_hash_salt="test-salt-16chars",
    )


@pytest_asyncio.fixture
async def test_db(tmp_path: Path) -> Database:
    db = Database(str(tmp_path / "test.db"))
    await db.connect()
    yield db
    await db.disconnect()


@pytest.fixture
def mock_asr() -> object:
    class MockASR:
        async def transcribe(self, audio_path: str, language: str) -> str:
            return "How do I register for NIN?"

    return MockASR()


@pytest.fixture
def mock_llm() -> object:
    class MockLLM:
        called = False

        async def generate(self, query, context, language, domain, is_out_of_knowledge) -> str:
            self.called = True
            if is_out_of_knowledge:
                return "I don't have that information."
            return "To register for NIN, visit your nearest NIMC office."

    return MockLLM()


@pytest.fixture
def mock_tts(tmp_path: Path) -> object:
    class MockTTS:
        async def synthesize(self, text: str, language: str) -> str:
            path = str(tmp_path / "response.mp3")
            Path(path).touch()
            return path

    return MockTTS()


@pytest.fixture
def intent_classifier() -> IntentClassifier:
    return IntentClassifier()
