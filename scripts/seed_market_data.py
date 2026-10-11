#!/usr/bin/env python3
"""Seed SQLite database with market prices from JSON."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import aiosqlite


async def seed(db_path: str, json_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    data_file = Path(json_path)
    if not data_file.exists():
        print(f"Seed file not found: {json_path}")
        sys.exit(1)

    prices = json.loads(data_file.read_text(encoding="utf-8"))
    print(f"Loaded {len(prices)} market price entries")

    async with aiosqlite.connect(db_path) as db:
        await db.execute("PRAGMA journal_mode=WAL")

        # Ensure table exists
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS market_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                commodity TEXT NOT NULL,
                market_name TEXT NOT NULL,
                state TEXT NOT NULL,
                price REAL NOT NULL,
                unit TEXT NOT NULL,
                source TEXT DEFAULT 'crowd',
                reported_by TEXT,
                reported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(commodity, market_name, state, unit)
            )
            """
        )

        inserted = 0
        for entry in prices:
            try:
                await db.execute(
                    """
                    INSERT INTO market_prices (commodity, market_name, state, price, unit, source, reported_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(commodity, market_name, state, unit) DO UPDATE SET
                        price = excluded.price,
                        source = excluded.source,
                        reported_at = excluded.reported_at
                    """,
                    (
                        entry["commodity"],
                        entry["market_name"],
                        entry["state"],
                        entry["price"],
                        entry["unit"],
                        entry.get("source", "seed"),
                        entry.get("reported_at", None),
                    ),
                )
                inserted += 1
            except Exception as e:
                print(f"Skipped entry: {e}")

        await db.commit()
        print(f"Seeded {inserted} market price entries into {db_path}")


if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else "data/hotline.db"
    json_file = sys.argv[2] if len(sys.argv) > 2 else "knowledge/market/seed_prices.json"
    asyncio.run(seed(db, json_file))
