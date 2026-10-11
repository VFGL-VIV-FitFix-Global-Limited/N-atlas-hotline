#!/usr/bin/env python3
"""Manual end-to-end pipeline test — run with a test audio file."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.models.database import Database
from app.models.schemas import Channel, Language, PipelineRequest
from app.pipeline.asr import NatlasASR
from app.pipeline.intent import IntentClassifier
from app.pipeline.llm import NatlasLLM
from app.pipeline.orchestrator import PipelineOrchestrator
from app.pipeline.rag import RAGRetriever
from app.pipeline.tts import GoogleTTS


async def main() -> None:
    settings = Settings()

    # Test intent classifier standalone
    print("=== Testing Intent Classifier ===")
    classifier = IntentClassifier()
    test_queries = [
        ("How do I get my NIN?", "en"),
        ("I have fever and headache", "en"),
        ("How much is rice?", "en"),
        ("How to plant cassava?", "en"),
        ("Tell me a joke", "en"),
    ]
    for query, lang in test_queries:
        domain = classifier.classify(query, lang)
        print(f"  [{lang}] '{query}' → {domain.value}")

    # Test RAG retrieval
    print("\n=== Testing RAG Retrieval ===")
    rag = RAGRetriever(settings.faiss_index_path, settings.rag_confidence_threshold)
    for query, lang in test_queries[:4]:
        domain = classifier.classify(query, lang)
        result = await rag.retrieve(query, domain.value)
        print(f"  Query: '{query}'")
        print(f"  Domain: {domain.value} | Score: {result.top_score:.3f} | OOK: {result.is_out_of_knowledge}")
        if result.chunks:
            print(f"  Top chunk: {result.chunks[0][:100]}...")
        print()

    # Test TTS
    print("=== Testing TTS ===")
    tts = GoogleTTS()
    audio_path = await tts.synthesize("Hello, this is a test of N-ATLAS Hotline.", "en")
    print(f"  Generated audio: {audio_path} ({Path(audio_path).stat().st_size} bytes)")

    print("\n✅ Pipeline test complete!")


if __name__ == "__main__":
    asyncio.run(main())
