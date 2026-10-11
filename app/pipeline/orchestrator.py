from __future__ import annotations

import asyncio
import time
from pathlib import Path

import structlog

from app.config import Settings
from app.models.database import Database
from app.models.schemas import (
    Channel,
    Domain,
    InteractionRecord,
    Language,
    PipelineRequest,
    PipelineResult,
)
from app.pipeline.asr import ASRError, ASRService
from app.pipeline.intent import IntentClassifier
from app.pipeline.llm import LLMService
from app.pipeline.rag import Retriever
from app.pipeline.tts import TTSService
from app.utils.hashing import hash_phone

logger = structlog.get_logger(__name__)

ERROR_MESSAGES: dict[str, str] = {
    "en": "I'm sorry, I couldn't process that. Please try again.",
    "yo": "Ẹ má bínú, mi ò lè ṣe ìbéèrè yẹn. Ẹ jọ̀wọ́ ẹ gbìyànjú lẹ́ẹ̀kan si.",
    "ha": "Yi hakuri, ban iya aiwatar da wannan ba. Da fatan za a sake gwadawa.",
    "ig": "Ndo, enweghị m ike ịrụ nke ahụ. Biko nwaa ọzọ.",
}


class PipelineOrchestrator:
    """Central coordinator: ASR → Intent → RAG → LLM → TTS."""

    def __init__(
        self,
        asr: ASRService,
        intent_classifier: IntentClassifier,
        rag: Retriever,
        llm: LLMService,
        tts: TTSService,
        db: Database,
        config: Settings,
    ) -> None:
        self._asr = asr
        self._intent = intent_classifier
        self._rag = rag
        self._llm = llm
        self._tts = tts
        self._db = db
        self._config = config

    async def process(self, request: PipelineRequest) -> PipelineResult:
        start_time = time.monotonic()

        try:
            # 1. ASR: audio → text
            question_text = await self._asr.transcribe(
                request.audio_path, request.language.value
            )

            # 2. Intent classification
            domain = self._intent.classify(question_text, request.language.value)

            # 3. RAG retrieval
            retrieval = await self._rag.retrieve(question_text, domain.value)

            # 4. LLM generation
            context = "\n\n---\n\n".join(retrieval.chunks) if retrieval.chunks else ""
            response_text = await self._llm.generate(
                query=question_text,
                context=context,
                language=request.language.value,
                domain=domain,
                is_out_of_knowledge=retrieval.is_out_of_knowledge,
            )

            # 5. TTS: text → audio
            response_audio_path = await self._tts.synthesize(
                response_text, request.language.value
            )

            # 6. Clean up input audio (privacy)
            self._cleanup_audio(request.audio_path)

            processing_time = int((time.monotonic() - start_time) * 1000)

            result = PipelineResult(
                session_id=request.session_id,
                question_text=question_text,
                intent_domain=domain,
                response_text=response_text,
                response_audio_path=response_audio_path,
                rag_confidence=retrieval.top_score,
                out_of_knowledge=retrieval.is_out_of_knowledge,
                processing_time_ms=processing_time,
                retrieved_sources=retrieval.sources,
            )

            # 7. Log interaction (fire-and-forget — don't break user response)
            asyncio.create_task(self._log_interaction(request, result))

            return result

        except ASRError as e:
            logger.error("pipeline_asr_failed", error=str(e))
            return await self._error_response(request, start_time, "asr_error")
        except Exception as e:
            logger.error("pipeline_unexpected_error", error=str(e), exc_info=True)
            return await self._error_response(request, start_time, "unexpected_error")

    async def _error_response(
        self, request: PipelineRequest, start_time: float, error_type: str
    ) -> PipelineResult:
        """Return a graceful error response the user will hear."""
        error_msg = ERROR_MESSAGES.get(request.language.value, ERROR_MESSAGES["en"])

        try:
            audio_path = await self._tts.synthesize(error_msg, request.language.value)
        except Exception:
            audio_path = ""

        processing_time = int((time.monotonic() - start_time) * 1000)

        return PipelineResult(
            session_id=request.session_id,
            question_text="",
            intent_domain=Domain.GENERAL,
            response_text=error_msg,
            response_audio_path=audio_path,
            rag_confidence=0.0,
            out_of_knowledge=True,
            processing_time_ms=processing_time,
            retrieved_sources=[],
        )

    async def _log_interaction(
        self, request: PipelineRequest, result: PipelineResult
    ) -> None:
        """Log the interaction to the database. Failures are swallowed."""
        try:
            phone_hash = hash_phone(request.phone_number, self._config.phone_hash_salt)
            record = InteractionRecord(
                session_id=request.session_id,
                channel=request.channel,
                user_phone_hash=phone_hash,
                language=request.language,
                question_text=result.question_text,
                intent_domain=result.intent_domain,
                response_text=result.response_text,
                rag_confidence=result.rag_confidence,
                out_of_knowledge=result.out_of_knowledge,
                processing_time_ms=result.processing_time_ms,
                retrieved_sources=result.retrieved_sources,
            )
            await self._db.log_interaction(record)
        except Exception:
            logger.exception("interaction_logging_failed", session_id=request.session_id)

    def _cleanup_audio(self, audio_path: str) -> None:
        """Delete audio file after transcription (privacy)."""
        try:
            Path(audio_path).unlink(missing_ok=True)
        except OSError:
            logger.warning("cleanup_failed", path=audio_path)
