from app.models.database import Database
from app.models.schemas import Language


class SessionStore:
    """SQLite-backed session store for language preferences. Survives restarts."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def set_language(self, phone_hash: str, language: Language) -> None:
        await self._db.save_session_language(phone_hash, language)

    async def get_language(self, phone_hash: str) -> Language | None:
        return await self._db.get_session_language(phone_hash)
