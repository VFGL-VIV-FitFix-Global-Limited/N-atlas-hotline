# N-ATLAS Hotline Architecture

Voice-first assistant for Nigerians. Callers use a phone or WhatsApp voice note. The same backend transcribes speech with N-ATLAS ASR, retrieves curated Nigerian knowledge, generates an answer with N-ATLAS LLM, and speaks it back.

## Channels

```
Phone (Africa's Talking) ──┐
                           ├──► FastAPI ──► Pipeline ──► audio out
WhatsApp Cloud API ────────┘
```

- **Phone:** IVR language select → record (30s) → process → play audio → optional follow-up. Flash-and-callback places an outbound call so the user pays nothing.
- **WhatsApp:** Language buttons on first contact. Voice notes process in a background task so the webhook returns 200 quickly. Text messages get a “send a voice note” reply.

## Pipeline

1. **ASR** — multipart audio to N-ATLAS (`NatlasASR`). Timeouts and 5xx retry 3 times (1s / 2s / 4s). 4xx is not retried.
2. **Intent** — keyword match in the user’s language **and** English (code-switching). Domains: government, health, market, agriculture, general.
3. **RAG** — FAISS over chunked markdown. Domain metadata filters results. Market queries also read `market_prices` in SQLite.
4. **LLM** — N-ATLAS with the PRD system + RAG prompts. Health, government and agriculture never use general model knowledge when retrieval is weak; canned helpline fallbacks are returned instead.
5. **TTS** — gTTS in a thread pool. Unsupported languages fall back to English with a warning log.
6. **Log** — interaction row with hashed phone. Logging is fire-and-forget so a DB failure does not drop the user response.
7. **Privacy** — input audio is deleted after transcription.

## Data

- **SQLite (WAL):** `interactions`, `sessions` (language, 24h expiry), `market_prices`.
- **FAISS:** built by `scripts/index_knowledge.py` from `knowledge/`.
- Phone numbers are SHA-256 hashed with `PHONE_HASH_SALT`. Raw numbers are never stored.

## App wiring

`lifespan` in `app/main.py` opens the DB, HTTP client, RAG index and orchestrator. FastAPI `Depends()` injects them. ASR / LLM / TTS are Protocols so tests swap mocks.

## Ops

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /health` | public | Load balancer |
| `GET /stats` | `X-API-Key` | Usage counts |
| `POST /voice/*` | Africa's Talking | Phone IVR |
| `GET/POST /whatsapp/webhook` | verify token | WhatsApp |

Structured logs (`structlog`) include `request_id` from middleware. Rate limits are per client IP via slowapi.
