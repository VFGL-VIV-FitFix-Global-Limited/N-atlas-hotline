from pathlib import Path

from app.config import Settings
from app.models.database import Database
from app.models.schemas import Channel, Domain, Language, PipelineRequest
from app.pipeline.asr import ASRError
from app.pipeline.intent import IntentClassifier
from app.pipeline.orchestrator import PipelineOrchestrator
from app.models.schemas import RetrievalResult


class FakeRAG:
    def __init__(self, result: RetrievalResult) -> None:
        self._result = result

    async def retrieve(self, query: str, domain: str, top_k: int = 3) -> RetrievalResult:
        return self._result


class FailingASR:
    async def transcribe(self, audio_path: str, language: str) -> str:
        raise ASRError("boom")


async def test_happy_path(settings: Settings, test_db: Database, mock_asr, mock_llm, mock_tts, tmp_path: Path) -> None:
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF")

    rag = FakeRAG(
        RetrievalResult(
            chunks=["NIN registration is free at NIMC offices."],
            sources=["nin_registration.md"],
            top_score=0.8,
            is_out_of_knowledge=False,
        )
    )
    orch = PipelineOrchestrator(mock_asr, IntentClassifier(), rag, mock_llm, mock_tts, test_db, settings)
    result = await orch.process(
        PipelineRequest(
            session_id="s1",
            channel=Channel.PHONE,
            audio_path=str(audio),
            language=Language.ENGLISH,
            phone_number="+2348012345678",
        )
    )
    assert result.question_text == "How do I register for NIN?"
    assert result.intent_domain == Domain.GOVERNMENT
    assert "NIN" in result.response_text
    assert result.out_of_knowledge is False
    assert not audio.exists()


async def test_out_of_knowledge_sensitive_domain(
    settings: Settings, test_db: Database, mock_asr, mock_tts, tmp_path: Path
) -> None:
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF")

    class TrackingLLM:
        called = False

        async def generate(self, query, context, language, domain, is_out_of_knowledge) -> str:
            self.called = True
            return "should not be used"

    llm = TrackingLLM()
    rag = FakeRAG(RetrievalResult(chunks=[], sources=[], top_score=0.1, is_out_of_knowledge=True))

    from app.pipeline.llm import FALLBACK_MESSAGES, NatlasLLM

    class FallbackLLM(NatlasLLM):
        def __init__(self) -> None:
            pass

        async def generate(self, query, context, language, domain, is_out_of_knowledge) -> str:
            if is_out_of_knowledge:
                return FALLBACK_MESSAGES["government"]["en"]
            llm.called = True
            return "generated"

    orch = PipelineOrchestrator(mock_asr, IntentClassifier(), rag, FallbackLLM(), mock_tts, test_db, settings)
    result = await orch.process(
        PipelineRequest(
            session_id="s2",
            channel=Channel.WHATSAPP,
            audio_path=str(audio),
            language=Language.ENGLISH,
            phone_number="+2348012345678",
        )
    )
    assert result.out_of_knowledge is True
    assert "NIMC" in result.response_text
    assert llm.called is False


async def test_asr_failure_is_graceful(
    settings: Settings, test_db: Database, mock_llm, mock_tts, tmp_path: Path
) -> None:
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF")
    rag = FakeRAG(RetrievalResult())
    orch = PipelineOrchestrator(FailingASR(), IntentClassifier(), rag, mock_llm, mock_tts, test_db, settings)
    result = await orch.process(
        PipelineRequest(
            session_id="s3",
            channel=Channel.PHONE,
            audio_path=str(audio),
            language=Language.ENGLISH,
            phone_number="+2348012345678",
        )
    )
    assert "try again" in result.response_text.lower()
    assert result.out_of_knowledge is True


async def test_interaction_logged(
    settings: Settings, test_db: Database, mock_asr, mock_llm, mock_tts, tmp_path: Path
) -> None:
    import asyncio

    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF")
    rag = FakeRAG(
        RetrievalResult(chunks=["ctx"], sources=["nin.md"], top_score=0.9, is_out_of_knowledge=False)
    )
    orch = PipelineOrchestrator(mock_asr, IntentClassifier(), rag, mock_llm, mock_tts, test_db, settings)
    await orch.process(
        PipelineRequest(
            session_id="s4",
            channel=Channel.PHONE,
            audio_path=str(audio),
            language=Language.ENGLISH,
            phone_number="+2348012345678",
        )
    )
    await asyncio.sleep(0.05)
    stats = await test_db.get_stats()
    assert stats.total_interactions == 1
    assert stats.by_channel.get("phone") == 1
