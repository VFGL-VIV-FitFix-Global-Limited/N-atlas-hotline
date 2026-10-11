#!/usr/bin/env python3
"""Build FAISS index from knowledge/ folder."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse YAML-style frontmatter from a markdown file."""
    metadata: dict[str, str] = {}
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if match:
        for line in match.group(1).strip().splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                metadata[key.strip()] = val.strip()
        text = text[match.end():]
    return metadata, text


def build_index(knowledge_path: str, output_path: str) -> None:
    knowledge_dir = Path(knowledge_path)
    if not knowledge_dir.exists():
        print(f"Knowledge directory not found: {knowledge_path}")
        sys.exit(1)

    # Collect all markdown files
    files = list(knowledge_dir.rglob("*.md"))
    if not files:
        print("No .md files found in knowledge directory")
        sys.exit(1)

    print(f"Found {len(files)} knowledge files")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )

    texts: list[str] = []
    metadatas: list[dict[str, str]] = []

    for file_path in sorted(files):
        raw = file_path.read_text(encoding="utf-8")
        frontmatter, content = parse_frontmatter(raw)

        # Determine domain from directory structure
        domain = file_path.parent.name

        chunks = splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append({
                "source": file_path.name,
                "domain": frontmatter.get("domain", domain),
                "topic": frontmatter.get("topic", file_path.stem),
                "chunk_index": str(i),
            })

    print(f"Created {len(texts)} chunks from {len(files)} files")

    # Build embeddings and FAISS index
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Building FAISS index...")
    index = FAISS.from_texts(texts, embedder, metadatas=metadatas)

    # Save index
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    index.save_local(output_path)
    print(f"Index saved to: {output_path}")
    print(f"Summary: {len(files)} files → {len(texts)} chunks indexed")


def verify_index(index_path: str) -> None:
    print(f"\nVerifying index at: {index_path}")
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    index = FAISS.load_local(index_path, embedder, allow_dangerous_deserialization=True)

    test_queries = [
        "How do I register for NIN?",
        "What are malaria symptoms?",
        "How to plant cassava?",
        "How much is rice in Lagos?",
    ]

    for query in test_queries:
        results = index.similarity_search_with_score(query, k=2)
        print(f"\nQuery: {query}")
        for doc, score in results:
            print(f"  Score: {score:.3f} | Source: {doc.metadata.get('source')} | Domain: {doc.metadata.get('domain')}")
            print(f"  Preview: {doc.page_content[:100]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from knowledge files")
    parser.add_argument("--knowledge", default="knowledge/", help="Path to knowledge directory")
    parser.add_argument("--output", default="data/faiss_index", help="Path to save FAISS index")
    parser.add_argument("--verify", action="store_true", help="Verify index after building")
    args = parser.parse_args()

    build_index(args.knowledge, args.output)

    if args.verify:
        verify_index(args.output)
