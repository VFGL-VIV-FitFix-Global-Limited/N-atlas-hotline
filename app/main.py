from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import httpx
import structlog
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
from app.pipeline.hf_models import LLM_HF_URL
from app.pipeline.intent import IntentClassifier
from app.pipeline.orchestrator import PipelineOrchestrator
from app.pipeline.rag import RAGRetriever
from app.pipeline.tts import GoogleTTS


def configure_logging(log_level: str) -> None:
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


async def _boot_pipeline() -> None:
    try:
        print("[boot] pipeline init starting", flush=True)
        settings = get_settings()
        configure_logging(settings.log_level)

        db = Database(settings.database_path)
        await db.connect()
        http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        asr = build_asr(http_client, settings)
        intent = IntentClassifier()
        rag = RAGRetriever(settings.faiss_index_path, settings.rag_confidence_threshold, db=db)
        llm = build_llm(http_client, settings)
        tts = GoogleTTS(settings.audio_dir)
        orchestrator = PipelineOrchestrator(asr, intent, rag, llm, tts, db, settings)

        dependencies._db = db
        dependencies._http_client = http_client
        dependencies._orchestrator = orchestrator
        print(f"[boot] pipeline ready  backend={settings.natlas_backend}", flush=True)
        await rag.warmup()
        print("[boot] rag warmup done", flush=True)
    except Exception as exc:
        print(f"[boot] PIPELINE INIT FAILED: {exc}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    print("[lifespan] yielding immediately", flush=True)
    boot = asyncio.create_task(_boot_pipeline())
    yield
    print("[lifespan] shutting down", flush=True)
    boot.cancel()
    http_client = getattr(dependencies, "_http_client", None)
    if http_client is not None:
        await http_client.aclose()
    db = dependencies._db
    if db is not None:
        await db.disconnect()


app = FastAPI(
    title="N-ATLAS Hotline",
    description="Voice-first AI assistant for Nigerians — phone call & WhatsApp",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(RequestIDMiddleware)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
register_error_handlers(app)

app.include_router(phone_router)
app.include_router(whatsapp_router)
app.include_router(twilio_router)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "version": "1.0.0",
        "service": "n-atlas-hotline",
        "ready": dependencies._orchestrator is not None,
    }


@app.get("/audio/{filename}")
async def serve_audio(filename: str, settings: Settings = Depends(get_settings)) -> FileResponse:
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
