from __future__ import annotations

import asyncio
import threading
from typing import Protocol

import structlog

from app.models.database import Database
from app.models.schemas import RetrievalResult

logger = structlog.get_logger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class Retriever(Protocol):
    async def retrieve(self, query: str, domain: str, top_k: int = 3) -> RetrievalResult: ...


class RAGRetriever:
    def __init__(
        self,
        index_path: str,
        threshold: float = 0.35,
        db: Database | None = None,
    ) -> None:
        self._index_path = index_path
        self._threshold = threshold
        self._db = db
        self._embedder = None
        self._index = None
        self._loaded = False
        self._load_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        """Load MiniLM/FAISS on a worker thread so uvicorn can bind first."""
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return

            from pathlib import Path

            index_dir = Path(self._index_path)
            if not index_dir.exists() or not any(index_dir.iterdir()):
                logger.warning("faiss_index_missing_skipping_embedder", path=self._index_path)
                self._loaded = True
                return

            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                from langchain_community.vectorstores import FAISS

                self._embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
                self._index = FAISS.load_local(
                    self._index_path,
                    self._embedder,
                    allow_dangerous_deserialization=True,
                )
                logger.info("faiss_index_loaded", path=self._index_path)
            except Exception:
                logger.warning("faiss_unavailable", path=self._index_path)
            self._loaded = True

    async def warmup(self) -> None:
        await asyncio.to_thread(self._ensure_loaded)

    async def retrieve(self, query: str, domain: str, top_k: int = 3) -> RetrievalResult:
        """Retrieve relevant chunks from the knowledge base."""
        if getattr(self, "_index_path", None):
            await self.warmup()
        if not self._index:
            logger.warning("rag_no_index")
            return RetrievalResult()

        try:
            results = self._index.similarity_search_with_score(query, k=top_k * 2)

            # Filter by domain metadata when available, keep top_k
            filtered = []
            for doc, score in results:
                doc_domain = doc.metadata.get("domain", "")
                if doc_domain == domain or not doc_domain:
                    filtered.append((doc, score))
                if len(filtered) >= top_k:
                    break

            # If domain filter was too strict, fall back to unfiltered top results
            if not filtered:
                filtered = results[:top_k]

            if not filtered:
                return RetrievalResult()

            chunks = [doc.page_content for doc, _ in filtered]
            sources = [doc.metadata.get("source", "unknown") for doc, _ in filtered]
            # FAISS returns L2 distance — lower is better. Convert to similarity.
            top_score = 1.0 / (1.0 + filtered[0][1]) if filtered else 0.0
            is_ook = top_score < self._threshold

            # For market domain, also query the database
            if domain == "market" and self._db:
                market_chunks = await self._get_market_chunks(query)
                if market_chunks:
                    chunks.extend(market_chunks)
                    sources.append("market_prices_db")
                    is_ook = False  # We have structured market data

            logger.info(
                "rag_retrieved",
                domain=domain,
                num_chunks=len(chunks),
                top_score=round(top_score, 3),
                out_of_knowledge=is_ook,
            )

            return RetrievalResult(
                chunks=chunks,
                sources=sources,
                top_score=top_score,
                is_out_of_knowledge=is_ook,
            )

        except Exception:
            logger.exception("rag_retrieval_failed")
            return RetrievalResult()

    async def _get_market_chunks(self, query: str) -> list[str]:
        """Query market prices DB and format as natural text."""
        if not self._db:
            return []

        try:
            prices = await self._db.get_market_prices()
            if not prices:
                return []

            lines = []
            for p in prices[:10]:
                lines.append(
                    f"{p.commodity} at {p.market_name} ({p.state}): ₦{p.price:,.0f} per {p.unit}"
                    + (f" (as of {p.reported_at})" if p.reported_at else "")
                )
            return ["\n".join(lines)] if lines else []

        except Exception:
            logger.exception("market_price_query_failed")
            return []
