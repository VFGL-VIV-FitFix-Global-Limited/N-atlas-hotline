from __future__ import annotations

import time


class PhoneRateLimiter:
    """In-memory per-phone-hash limiter. Process-local — fine for MVP."""

    def __init__(self) -> None:
        self._hits: dict[str, list[float]] = {}

    def allow(self, phone_hash: str, limit: int, window_seconds: float = 60.0) -> bool:
        now = time.monotonic()
        recent = [t for t in self._hits.get(phone_hash, []) if now - t < window_seconds]
        if len(recent) >= limit:
            self._hits[phone_hash] = recent
            return False
        recent.append(now)
        self._hits[phone_hash] = recent
        return True


phone_limiter = PhoneRateLimiter()
