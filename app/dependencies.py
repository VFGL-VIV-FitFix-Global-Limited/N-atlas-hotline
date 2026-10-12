from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, Header, HTTPException

from app.config import Settings
from app.models.database import Database
from app.pipeline.orchestrator import PipelineOrchestrator
from app.services.session import SessionStore

# Module-level singletons — set during lifespan startup
_db: Database | None = None
_orchestrator: PipelineOrchestrator | None = None
_http_client = None


@lru_cache
def get_settings() -> Settings:
    return Settings()


async def get_db() -> Database:
    if _db is None:
        raise HTTPException(status_code=503, detail="Service is still starting")
    return _db


async def get_orchestrator() -> PipelineOrchestrator:
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Service is still starting")
    return _orchestrator


async def get_session_store() -> SessionStore:
    db = await get_db()
    return SessionStore(db)


async def verify_api_key(
    x_api_key: str = Header(...),
    settings: Settings = Depends(get_settings),
) -> str:
    """Verify API key for protected endpoints."""
    if x_api_key != settings.api_secret_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return x_api_key
