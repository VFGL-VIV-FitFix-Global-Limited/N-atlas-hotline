import structlog

from app.models.database import Database
from app.models.schemas import InteractionRecord

logger = structlog.get_logger(__name__)


class InteractionLogger:
    """Logs pipeline interactions to the database."""

    def __init__(self, db: Database, salt: str) -> None:
        self._db = db
        self._salt = salt

    async def log(self, record: InteractionRecord) -> int:
        try:
            row_id = await self._db.log_interaction(record)
            logger.info(
                "interaction_logged",
                session_id=record.session_id,
                channel=record.channel.value,
                domain=record.intent_domain.value,
                language=record.language.value,
                out_of_knowledge=record.out_of_knowledge,
                processing_time_ms=record.processing_time_ms,
            )
            return row_id
        except Exception:
            logger.exception("interaction_log_failed", session_id=record.session_id)
            return 0
