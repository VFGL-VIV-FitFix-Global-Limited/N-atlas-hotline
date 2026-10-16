# N-ATLAS Hotline Architecture

Voice-first assistant for Nigerians. **Canonical product description:** [`docs/prd.md`](prd.md) (use that for NAIC write-ups).

## Channels (as built)

```
Hugging Face Space (evaluators) ──► ZeroGPU: NCAIR ASR + N-ATLaS LLM + Soro/gTTS
                                          ▲
Twilio WhatsApp ──► Railway ── NATLAS_BACKEND=space ──┘
Phone (Africa's Talking) ──► Railway /voice/* (AT credentials required)
Optional Colab gateway ──► NATLAS_BACKEND=gateway (filming only)
```

- **Space:** self-serve Gradio. Primary judge URL. API `/ask` for WhatsApp.
- **WhatsApp:** Twilio form webhooks. One outbound message with N-ATLaS text + spoken audio. `NATLAS_BACKEND=space` uses the Space GPU (no Colab).
- **Phone:** IVR language select → record (30s) → process → play audio → follow-up. Flash (duration 0 hangup) places an outbound callback.

## Pipeline (gateway / local)

1. **ASR** : NCAIR Whisper Small via gateway HTTP or local `pipeline`.
2. **Intent** : keywords in the user language and English.
3. **RAG** : FAISS over `knowledge/` when the index exists; production may skip it.
4. **LLM** : `NCAIR1/N-ATLaS` only on the demo path.
5. **TTS** : Soro (yo/ha/ig), gTTS (en). Not N-ATLaS.
6. **Log** ; SQLite + JSONL; hashed phone; input audio deleted.

## Ops

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /health` | public | Load balancer |
| `GET /stats` | `X-API-Key` | Counts |
| `GET /interactions` | `X-API-Key` | Q&A JSON |
| `GET /interactions.csv` | `X-API-Key` | Q&A CSV |
| `POST /twilio/webhook` | Twilio | WhatsApp |
| `GET/POST /whatsapp/webhook` | verify token | Meta Cloud API |
| `POST /voice/*` | Africa's Talking | Phone IVR |
