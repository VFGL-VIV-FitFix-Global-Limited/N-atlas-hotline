# N-ATLAS Hotline

Voice-first AI assistant for Nigerians. Call or send a WhatsApp voice note in Yoruba, Hausa, Igbo or English. Answers cover government services, health, agriculture and market prices.

Powered by official **[NCAIR1/N-ATLaS](https://huggingface.co/NCAIR1/N-ATLaS)** (ASR + text LLM). Spoken replies use **Soro-TTS** (Yoruba/Igbo/Hausa) or **gTTS** (English) — N-ATLaS has no TTS.

Evaluator demo: [Hugging Face Space](https://huggingface.co/spaces/onomeakpobaro/hotline). WhatsApp on Railway uses `NATLAS_BACKEND=space` so it hits that same GPU. Product write-up: [`docs/prd.md`](docs/prd.md).

**VFGL Tech** · NAIC 2026 · PS2 Voice-First Access

## Requirements

- Python 3.11+
- ffmpeg (audio conversion)
- N-ATLAS, Africa's Talking, and WhatsApp Cloud API credentials

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env   # fill in keys
mkdir -p data
python scripts/index_knowledge.py --verify
python scripts/seed_market_data.py
uvicorn app.main:app --reload --port 8000
```

Health check: `GET http://localhost:8000/health`  
Stats: `GET /stats`. Q&A log: `GET /interactions` and `GET /interactions.csv` (header `X-API-Key`).

## Webhooks

Point Africa's Talking voice callback to `/voice/callback` and missed-call/flash events to `/voice/flash`.  
Point WhatsApp to `/whatsapp/webhook` (GET verify + POST messages). Set `WA_APP_SECRET` so POSTs are HMAC-checked.

After deploy, `APP_BASE_URL` must be public HTTPS — Africa's Talking `<Play>` and WhatsApp fetch `/audio/{file}`.

## Tests

```bash
pytest
```

## Docker

```bash
docker compose up --build
```

## Project layout

See `docs/prd.md` (as-built product), `docs/architecture.md`, and `docs/natlas_integration.md`.
