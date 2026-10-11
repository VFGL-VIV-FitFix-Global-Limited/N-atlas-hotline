from app.models.database import Database
from app.models.schemas import Language
from app.services.session import SessionStore


async def test_set_and_get_language(test_db: Database) -> None:
    store = SessionStore(test_db)
    await store.set_language("abc123", Language.HAUSA)
    assert await store.get_language("abc123") == Language.HAUSA


async def test_unknown_session_returns_none(test_db: Database) -> None:
    store = SessionStore(test_db)
    assert await store.get_language("missing") is None


async def test_language_can_be_updated(test_db: Database) -> None:
    store = SessionStore(test_db)
    await store.set_language("abc123", Language.YORUBA)
    await store.set_language("abc123", Language.IGBO)
    assert await store.get_language("abc123") == Language.IGBO
