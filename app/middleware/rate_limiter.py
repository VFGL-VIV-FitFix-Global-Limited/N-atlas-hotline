from __future__ import annotations

import time

from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request
from starlette.responses import JSONResponse


def _get_client_host(request: Request) -> str:
    return request.client.host if request.client else "unknown"


limiter = Limiter(key_func=_get_client_host)


async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": "Please wait a moment before trying again.",
        },
    )
