-- interactions table
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    user_phone_hash TEXT,
    language TEXT NOT NULL,
    question_text TEXT,
    intent_domain TEXT,
    retrieved_sources TEXT,
    response_text TEXT,
    rag_confidence REAL,
    out_of_knowledge BOOLEAN DEFAULT 0,
    processing_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- market prices table
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
);

-- session language preferences
CREATE TABLE IF NOT EXISTS sessions (
    phone_hash TEXT PRIMARY KEY,
    language TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- indexes for common queries
CREATE INDEX IF NOT EXISTS idx_interactions_channel ON interactions(channel);
CREATE INDEX IF NOT EXISTS idx_interactions_language ON interactions(language);
CREATE INDEX IF NOT EXISTS idx_interactions_domain ON interactions(intent_domain);
CREATE INDEX IF NOT EXISTS idx_interactions_created ON interactions(created_at);
CREATE INDEX IF NOT EXISTS idx_market_commodity ON market_prices(commodity);
CREATE INDEX IF NOT EXISTS idx_market_state ON market_prices(state);
