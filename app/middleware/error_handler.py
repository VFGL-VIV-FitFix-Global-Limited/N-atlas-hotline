import httpx
import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.pipeline.asr import ASRError

logger = structlog.get_logger(__name__)


async def asr_error_handler(request: Request, exc: ASRError) -> JSONResponse:
    logger.error("asr_error", error=exc.message, status_code=exc.status_code)
    return JSONResponse(
        status_code=502,
        content={"error": "asr_failed", "message": "Voice processing failed. Please try again."},
    )


async def timeout_error_handler(request: Request, exc: httpx.TimeoutException) -> JSONResponse:
    logger.error("timeout_error", url=str(getattr(exc, "request", {}).url if hasattr(exc, "request") else "unknown"))
    return JSONResponse(
        status_code=504,
        content={"error": "timeout", "message": "The service took too long. Please try again."},
    )


async def general_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_error", error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "message": "Something went wrong. Please try again."},
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register all global exception handlers."""
    app.add_exception_handler(ASRError, asr_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(httpx.TimeoutException, timeout_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, general_error_handler)  # type: ignore[arg-type]
