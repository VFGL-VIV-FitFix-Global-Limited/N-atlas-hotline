from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx
import structlog
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from slowapi.errors import RateLimitExceeded

from app import dependencies
from app.channels.phone import router as phone_router
from app.channels.twilio_whatsapp import router as twilio_router
from app.channels.whatsapp import router as whatsapp_router
from app.config import Settings
from app.dependencies import get_db, get_settings, verify_api_key
from app.middleware.error_handler import register_error_handlers
from app.middleware.rate_limiter import limiter, rate_limit_handler
from app.middleware.request_id import RequestIDMiddleware
from app.models.database import Database
from app.models.schemas import StatsResponse
from app.pipeline.factory import build_asr, build_llm
from app.pipeline.hf_models import ATTRIBUTION, LLM_HF_URL
from app.pipeline.intent import IntentClassifier
from app.pipeline.orchestrator import PipelineOrchestrator
from app.pipeline.rag import RAGRetriever
from app.pipeline.tts import GoogleTTS


def configure_logging(log_level: str) -> None:
    """Configure structlog for JSON-formatted structured logging."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper(), logging.INFO),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: wire all dependencies. Shutdown: clean up."""
    settings = get_settings()
    configure_logging(settings.log_level)

    logger = structlog.get_logger(__name__)
    logger.info("starting_up", environment=settings.environment.value)

    # Database
    db = Database(settings.database_path)
    await db.connect()

    # HTTP client for external API calls
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))

    # Pipeline components
    asr = build_asr(http_client, settings)
    intent = IntentClassifier()
    rag = RAGRetriever(settings.faiss_index_path, settings.rag_confidence_threshold, db=db)
    llm = build_llm(http_client, settings)
    tts = GoogleTTS(settings.audio_dir)
    orchestrator = PipelineOrchestrator(asr, intent, rag, llm, tts, db, settings)

    # Set module-level singletons for dependency injection
    dependencies._db = db
    dependencies._orchestrator = orchestrator

    logger.info(
        "startup_complete",
        natlas_backend=settings.natlas_backend,
        natlas_model=settings.natlas_model,
        source=LLM_HF_URL,
    )
    yield

    # Shutdown
    logger.info("shutting_down")
    await http_client.aclose()
    await db.disconnect()
    logger.info("shutdown_complete")


app = FastAPI(
    title="N-ATLAS Hotline",
    description="Voice-first AI assistant for Nigerians — phone call & WhatsApp",
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(RequestIDMiddleware)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
register_error_handlers(app)

# Routers
app.include_router(phone_router)
app.include_router(whatsapp_router)
app.include_router(twilio_router)


# --- Health & Stats ---

@app.get("/health")
async def health(settings: Settings = Depends(get_settings)) -> dict:
    natlas: dict = {
        "backend": settings.natlas_backend,
        "model": settings.natlas_model,
        "source": LLM_HF_URL,
        "attribution": ATTRIBUTION,
    }
    if settings.natlas_backend == "gateway" and settings.natlas_base_url:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(settings.natlas_health_url, timeout=5.0)
                natlas["gateway"] = (
                    resp.json()
                    if "application/json" in resp.headers.get("content-type", "")
                    else {"http_status": resp.status_code}
                )
        except Exception as exc:
            natlas["gateway"] = {"status": "unreachable", "error": type(exc).__name__}

    return {
        "status": "ok",
        "version": "1.0.0",
        "service": "n-atlas-hotline",
        "natlas": natlas,
    }


@app.get("/audio/{filename}")
async def serve_audio(filename: str, settings: Settings = Depends(get_settings)) -> FileResponse:
    """Public URL for Africa's Talking <Play> and WhatsApp audio.link."""
    safe = Path(filename).name
    if safe != filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = Path(settings.audio_dir) / safe
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Audio not found")
    media = "audio/ogg" if path.suffix == ".ogg" else "audio/mpeg"
    return FileResponse(path, media_type=media)


@app.get("/stats", dependencies=[Depends(verify_api_key)])
async def stats(db: Database = Depends(get_db)) -> StatsResponse:
    return await db.get_stats()
