from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import structlog

from app.models.schemas import (
    InteractionRecord,
    Language,
    MarketPrice,
    StatsResponse,
)

logger = structlog.get_logger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent.parent / "migrations"


class Database:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._jsonl_path = Path(db_path).parent / "interactions.jsonl"
        self._connection: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = await aiosqlite.connect(self._db_path)
        self._connection.row_factory = aiosqlite.Row
        await self._connection.execute("PRAGMA journal_mode=WAL")
        await self._connection.execute("PRAGMA foreign_keys=ON")
        await self._run_migrations()
        logger.info("database_connected", path=self._db_path)

    async def disconnect(self) -> None:
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.info("database_disconnected")

    @property
    def conn(self) -> aiosqlite.Connection:
        assert self._connection is not None, "Database not connected — call connect() first"
        return self._connection

    async def _run_migrations(self) -> None:
        migration_file = MIGRATIONS_DIR / "001_initial_schema.sql"
        if migration_file.exists():
            sql = migration_file.read_text()
            await self.conn.executescript(sql)
            logger.info("migrations_applied", file=migration_file.name)

    # Interactions 

    async def log_interaction(self, record: InteractionRecord) -> int:
        cursor = await self.conn.execute(
            """
            INSERT INTO interactions
                (session_id, channel, user_phone_hash, language, question_text,
                 intent_domain, retrieved_sources, response_text, rag_confidence,
                 out_of_knowledge, processing_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.session_id,
                record.channel.value,
                record.user_phone_hash,
                record.language.value,
                record.question_text,
                record.intent_domain.value,
                json.dumps(record.retrieved_sources),
                record.response_text,
                record.rag_confidence,
                record.out_of_knowledge,
                record.processing_time_ms,
            ),
        )
        await self.conn.commit()
        row_id = cursor.lastrowid or 0
        self._append_jsonl(record, row_id)
        return row_id

    def _append_jsonl(self, record: InteractionRecord, row_id: int) -> None:
        """Durable-enough copy next to SQLite so you can download Q&A for submission."""
        try:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(
                {
                    "id": row_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "session_id": record.session_id,
                    "channel": record.channel.value,
                    "language": record.language.value,
                    "question_text": record.question_text,
                    "response_text": record.response_text,
                    "intent_domain": record.intent_domain.value,
                    "rag_confidence": record.rag_confidence,
                    "out_of_knowledge": record.out_of_knowledge,
                    "processing_time_ms": record.processing_time_ms,
                    "retrieved_sources": record.retrieved_sources,
                },
                ensure_ascii=False,
            )
            with self._jsonl_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            logger.warning("interaction_jsonl_write_failed", path=str(self._jsonl_path))

    async def list_interactions(self, limit: int = 500) -> list[dict]:
        cap = max(1, min(limit, 2000))
        rows = await (
            await self.conn.execute(
                """
                SELECT id, created_at, session_id, channel, language,
                       question_text, response_text, intent_domain,
                       rag_confidence, out_of_knowledge, processing_time_ms,
                       retrieved_sources
                FROM interactions
                ORDER BY id DESC
                LIMIT ?
                """,
                (cap,),
            )
        ).fetchall()
        out: list[dict] = []
        for r in rows:
            sources = r["retrieved_sources"]
            try:
                sources = json.loads(sources) if sources else []
            except json.JSONDecodeError:
                sources = []
            out.append(
                {
                    "id": r["id"],
                    "created_at": r["created_at"],
                    "session_id": r["session_id"],
                    "channel": r["channel"],
                    "language": r["language"],
                    "question_text": r["question_text"],
                    "response_text": r["response_text"],
                    "intent_domain": r["intent_domain"],
                    "rag_confidence": r["rag_confidence"],
                    "out_of_knowledge": bool(r["out_of_knowledge"]),
                    "processing_time_ms": r["processing_time_ms"],
                    "retrieved_sources": sources,
                }
            )
        return out

    async def get_stats(self) -> StatsResponse:
        # Total
        row = await (await self.conn.execute("SELECT COUNT(*) as cnt FROM interactions")).fetchone()
        total = row["cnt"] if row else 0

        # By channel
        by_channel: dict[str, int] = {}
        async for r in await self.conn.execute(
            "SELECT channel, COUNT(*) as cnt FROM interactions GROUP BY channel"
        ):
            by_channel[r["channel"]] = r["cnt"]

        # By language
        by_language: dict[str, int] = {}
        async for r in await self.conn.execute(
            "SELECT language, COUNT(*) as cnt FROM interactions GROUP BY language"
        ):
            by_language[r["language"]] = r["cnt"]

        # By domain
        by_domain: dict[str, int] = {}
        async for r in await self.conn.execute(
            "SELECT intent_domain, COUNT(*) as cnt FROM interactions GROUP BY intent_domain"
        ):
            by_domain[r["intent_domain"]] = r["cnt"]

        # Out of knowledge
        row = await (
            await self.conn.execute("SELECT COUNT(*) as cnt FROM interactions WHERE out_of_knowledge = 1")
        ).fetchone()
        ook_count = row["cnt"] if row else 0

        # Average response time
        row = await (
            await self.conn.execute("SELECT AVG(processing_time_ms) as avg_ms FROM interactions")
        ).fetchone()
        avg_ms = row["avg_ms"] if row and row["avg_ms"] else 0.0

        return StatsResponse(
            total_interactions=total,
            by_channel=by_channel,
            by_language=by_language,
            by_domain=by_domain,
            out_of_knowledge_count=ook_count,
            avg_response_time_ms=round(avg_ms, 1),
        )

    # Sessions

    async def save_session_language(self, phone_hash: str, language: Language) -> None:
        await self.conn.execute(
            """
            INSERT INTO sessions (phone_hash, language, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(phone_hash) DO UPDATE SET language = excluded.language, updated_at = CURRENT_TIMESTAMP
            """,
            (phone_hash, language.value),
        )
        await self.conn.commit()

    async def get_session_language(self, phone_hash: str) -> Language | None:
        row = await (
            await self.conn.execute(
                """
                SELECT language FROM sessions
                WHERE phone_hash = ?
                  AND updated_at > datetime('now', '-24 hours')
                """,
                (phone_hash,),
            )
        ).fetchone()
        if row:
            return Language(row["language"])
        return None

    # Market Prices 

    async def get_market_prices(
        self, commodity: str | None = None, state: str | None = None
    ) -> list[MarketPrice]:
        query = "SELECT * FROM market_prices WHERE 1=1"
        params: list[str] = []

        if commodity:
            query += " AND LOWER(commodity) LIKE ?"
            params.append(f"%{commodity.lower()}%")
        if state:
            query += " AND LOWER(state) LIKE ?"
            params.append(f"%{state.lower()}%")

        query += " ORDER BY reported_at DESC LIMIT 20"

        rows = await (await self.conn.execute(query, params)).fetchall()
        return [
            MarketPrice(
                commodity=r["commodity"],
                market_name=r["market_name"],
                state=r["state"],
                price=r["price"],
                unit=r["unit"],
                source=r["source"],
                reported_at=r["reported_at"],
            )
            for r in rows
        ]
