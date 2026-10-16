# N-ATLAS Hotline — Product Requirements Document (as built)

**Product:** N-ATLAS Hotline  
**Organisation:** VFGL (VFGL Tech)  
**Challenge:** NAIC 2026 — PS2 Voice-First Access  
**Document status:** Implementation-aligned (October 2026)  
**Audience:** Submission writers (Claude / humans). Use this over older architecture notes if they conflict.

N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy, powered by Awarri Technologies.

---

## 1. One-line pitch

A voice-first hotline so Nigerians can **speak** Yoruba, Hausa, Igbo, or Nigerian English, be **heard** by official NCAIR ASR, **answered** by official `NCAIR1/N-ATLaS`, and **hear a reply** in that language.

## 2. Problem

Many Nigerians cannot reliably use typed English web UIs for government, health, farming, and market questions. Literacy, language, and data cost block access. A phone call or WhatsApp voice note is the native interface.

## 3. Users

| Who | How they use it | What we ship for them |
|---|---|---|
| **NAIC evaluators** | Browser, no Twilio join | Hugging Face Space (self-serve) |
| **End users** | WhatsApp voice note, later phone / flash | Railway + Twilio (and Africa’s Talking when live) |
| **Operators** | Usage and Q&A export | `/stats`, `/interactions`, `/interactions.csv` |

## 4. Non-negotiable product rules

1. **Hear and think are official N-ATLaS only.** Four NCAIR ASR repos + `NCAIR1/N-ATLaS` (Llama-3 8B). No Whisper-large, Qwen, GPT, or Gemini on the reasoning/transcription path of the demo.
2. **N-ATLaS does not speak.** The official drop is four ASR models + one text LLM. There is no NCAIR TTS.
3. **Spoken replies are labelled as a renderer.** Yoruba / Igbo / Hausa: Soro-TTS MMS/VITS (`Shinzmann/soro-tts-yor`, `soro-tts-ibo`, `soro-tts-hau`, CC BY-NC). English: gTTS.
4. **Do not clone other NAIC teams’ gateways.** GPU serving is our Space, our Colab notebook, or a host we operate.
5. **Privacy:** raw phone numbers are hashed (`PHONE_HASH_SALT`). Input voice notes are deleted after transcription. Logs store **text** Q&A, not audio.

## 5. Official N-ATLaS stack (what “powered by N-ATLaS” means)

| Role | Model | Task |
|---|---|---|
| Hear English | `NCAIR1/NigerianAccentedEnglish` | ASR (Whisper Small) |
| Hear Yoruba | `NCAIR1/Yoruba-ASR` | ASR |
| Hear Hausa | `NCAIR1/Hausa-ASR` | ASR |
| Hear Igbo | `NCAIR1/Igbo-ASR` | ASR |
| Answer | `NCAIR1/N-ATLaS` | Text LLM (`from_pretrained`, `apply_chat_template` + `date_string`, `generate`) |

NCAIR’s own terms: N-ATLaS consists of **four ASR models and one text LLM**. Marketing language that “N-ATLaS speaks Hausa…” means the **LLM writes text** in those languages.

## 6. What is implemented

### 6.1 Hugging Face Space — evaluator demo (primary submission URL)

- **URL:** https://huggingface.co/spaces/onomeakpobaro/hotline  
- **Runtime:** Gradio 5.9.1, ZeroGPU (`@spaces.GPU`, duration 120s). No `.cuda()` at import.  
- **Flow:** Record/upload voice note → language radio (English / Yoruba / Hausa / Igbo) → **Ask N-ATLaS**.  
- **Outputs:** “What we heard (NCAIR ASR)”, “N-ATLaS reply”, spoken audio labelled *not N-ATLaS — Soro-TTS / gTTS*.  
- **API for WhatsApp:** Gradio endpoint `/ask` (`api_name="ask"`), with a schema patch so Gradio 5 `additionalProperties: true` does not crash `get_api_info`.  
- **Secrets:** `HF_TOKEN` (gated NCAIR weights). Optional `HOTLINE_LOG_URL` + `HOTLINE_LOG_KEY` to POST Q&A into Railway’s log.  
- **Quota:** ZeroGPU is per Hugging Face account per day. Guests have little time; signed-in free accounts more; PRO more. “Quota exceeded” is not an app crash.

### 6.2 Railway FastAPI — user channels

- **App:** FastAPI + Uvicorn, Docker, `PORT` from Railway (no hardcoded `EXPOSE` conflict).  
- **Health:** `GET /health` (`ready` once the pipeline has booted in a background task so the port binds immediately).  
- **WhatsApp (live path):** Twilio WhatsApp sandbox/webhook `POST /twilio/webhook`. Voice notes download with retries on Twilio media 404. Webhook returns 200; work runs in a background task. Sandbox **429 is not retried** (retries made the cap worse).  
- **When `NATLAS_BACKEND=space`:** Railway does **not** run the 8B. It sends the voice note to the Space `/ask` via `gradio_client` and returns transcript + N-ATLaS text + Soro/gTTS audio in **one** Twilio message (`Body` + `MediaUrl`).  
- **When `NATLAS_BACKEND=gateway`:** Railway POSTs to a Colab/GPU host `NATLAS_BASE_URL` (`/v1/audio/transcriptions`, `/v1/chat/completions`). Trycloudflare often **524s** (~100s) if the 8B is CPU-offloaded. This is a demo/filming path, not 24/7.  
- **WhatsApp Cloud API** (`/whatsapp/webhook`) still exists in code; production user testing has been **Twilio**.  
- **Phone / flash (coded, ops-dependent):** Africa’s Talking IVR at `/voice/callback`, language DTMF 1–4, 30s record, play answer, follow-up. Hangup with duration 0 on a Nigerian number triggers outbound callback (user does not pay airtime). Needs AT Voice product, number, and outbound credit. Toll-free is an AT/NCC number, not a route. IVR menus still use AT `<Say>`; **answers** should be `<Play>` of generated audio.

### 6.3 Pipeline (gateway / local / huggingface backends)

Order: ASR → intent (keyword, user language + English) → RAG (FAISS; **skipped in production if `data/faiss_index` missing**) → LLM (system + RAG prompts; health/government/agriculture use canned fallbacks if retrieval is weak) → TTS → log → delete input audio.

**TTS (`HotlineTTS`):** Soro on CPU for `yo` / `ha` / `ig`; gTTS for English; gTTS fallback if Soro fails. Igbo gTTS was never supported; Soro closed that gap.

**LLM prompts:** spoken-style, under ~150 words, always reply in the selected language. Health: general guidance, see a doctor. Government: not legal advice.

**Backends (`NATLAS_BACKEND`):**

| Value | What runs |
|---|---|
| `space` | WhatsApp → Hugging Face Space (same GPU as reviewers). Preferred for demo. |
| `gateway` | HTTP to Colab / OpenAI-compatible GPU (`NATLAS_BASE_URL` + `NATLAS_API_KEY`) |
| `huggingface` | HF Inference HTTP (serverless **does not host NCAIR1**; needs dedicated endpoints) |
| `local` | `from_pretrained` on a GPU box. Not for Railway CPU |

### 6.4 Knowledge

Markdown under `knowledge/` for government, health, agriculture, market. Index via `scripts/index_knowledge.py`. Market seed via `scripts/seed_market_data.py`. Production Railway image may run **without** FAISS (`faiss_index_missing_skipping_embedder`); the LLM still answers.

### 6.5 Interaction log (for submission evidence)

After a successful turn, Railway writes:

- SQLite `data/hotline.db` table `interactions`  
- Append-only `data/interactions.jsonl`  

Fields: timestamp, channel (`whatsapp` / `phone` / `space`), language, **question_text**, **response_text**, intent, RAG score, processing time. Phone stored as hash only. **No voice files.**

Export (header `X-API-Key: API_SECRET_KEY`):

- `GET /stats` — counts  
- `GET /interactions` — JSON Q&A  
- `GET /interactions.csv` — spreadsheet for NAIC  
- `POST /interactions` — Space (and other UIs) ingest the same log  

**Redeploys can wipe the Railway disk.** Download CSV before deploy. Space tests appear in this log only if `HOTLINE_LOG_URL` / `HOTLINE_LOG_KEY` are set on the Space.

### 6.6 Privacy and safety

- Rate limits (IP + phone limiter on voice).  
- Twilio signature validation when auth token is set.  
- WhatsApp Cloud HMAC when `WA_APP_SECRET` is set.  
- Health/government: no diagnosis, no legal advice; helpline fallbacks in four languages.

## 7. What is explicitly not in scope / not done

| Item | Status |
|---|---|
| Official NCAIR TTS | Does not exist |
| 24/7 Colab | Not possible on free Colab; trycloudflare 524 + CPU offload |
| Competitor N-ATLAS Kit / Modal URL | Must not be used |
| Production Meta WhatsApp sender | Code exists; live tests used Twilio sandbox |
| Africa’s Talking production flash/toll-free | Code exists; needs AT wallet + numbers |
| Marketing landing page | Not built; Space is the brochure + demo |
| Storing user voice recordings | Intentionally not stored |

## 8. UX copy (honest, reuse in slides/PDF)

- Hear: official NCAIR ASR.  
- Answer: official `NCAIR1/N-ATLaS`.  
- Speak: **not N-ATLaS** — Soro-TTS (Yoruba/Igbo/Hausa) or gTTS (English).  
- Evaluator URL: Hugging Face Space.  
- User channel: WhatsApp voice note (same models when `NATLAS_BACKEND=space`).  
- First Space request: 1–2 minutes (GPU + weights). Short voice notes. Sign in to Hugging Face for quota.

## 9. System diagram (as built)

```
Evaluator  →  Hugging Face Space (ZeroGPU)
                 ASR NCAIR ×4  +  LLM NCAIR1/N-ATLaS  +  Soro/gTTS
                 Gradio UI  +  API /ask

User WhatsApp  →  Twilio  →  Railway FastAPI
                 NATLAS_BACKEND=space  →  Space /ask  →  one WA message (text + audio)

Optional filming  →  Colab GPU + cloudflared  ←  NATLAS_BACKEND=gateway
                 (fragile; Cloudflare ~100s 524 if 8B is CPU-offloaded)

Phone / flash  →  Africa’s Talking  →  Railway /voice/*
                 (implemented; production number/credit required)
```

## 10. Key endpoints (Railway)

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | public | Load balancer / ready flag |
| GET | `/audio/{file}` | public | Twilio/AT fetch TTS |
| GET | `/stats` | `X-API-Key` | Counts |
| GET | `/interactions` | `X-API-Key` | Q&A JSON |
| GET | `/interactions.csv` | `X-API-Key` | Q&A CSV |
| POST | `/interactions` | `X-API-Key` | Ingest Space (or other) Q&A |
| POST | `/twilio/webhook` | Twilio signature | WhatsApp |
| GET/POST | `/whatsapp/webhook` | verify + HMAC | Meta Cloud API |
| POST | `/voice/callback` etc. | Africa’s Talking | Phone IVR |

## 11. Environment (operators)

Important variables: `NATLAS_BACKEND`, `HF_SPACE_ID` (default `onomeakpobaro/hotline`), `HF_TOKEN`, `NATLAS_BASE_URL` / `NATLAS_API_KEY` (gateway only), `TWILIO_*`, `AT_*`, `APP_BASE_URL`, `API_SECRET_KEY`, `PHONE_HASH_SALT`.

If `NATLAS_BASE_URL` is set and ASR/LLM URLs are still the `api.natlas.ng` defaults, the app rewrites them to `/v1/audio/transcriptions` and `/v1/chat/completions`. Dead Colab tunnels must be removed from Railway.

## 12. Known limitations (write these into the submission, do not hide)

1. ZeroGPU daily quota; evaluators should be signed in.  
2. Soro needs tone marks; N-ATLaS often omits diacritics → flatter Yoruba/Igbo.  
3. Soro MMS is CC BY-NC (research/demo, not unrestricted commercial).  
4. Twilio sandbox rate limits (429).  
5. Railway SQLite is not a durable archive across deploys.  
6. Colab+tunnel is unsuitable as the standing production GPU.  
7. RAG index may be absent on Railway; answers still come from N-ATLaS.

## 13. Suggested artefacts for NAIC documents

- **Live demo:** Space URL.  
- **Architecture / PRD:** this file.  
- **Evidence of use:** WhatsApp screenshots + `interactions.csv` pulled before redeploy.  
- **Video:** user speaks → transcript → N-ATLaS text → Soro playback; one line that N-ATLaS has no TTS.  
- **Attribution** on every page: Federal Ministry of Communications, Innovation and Digital Economy; Awarri Technologies; NCAIR.

## 14. Repo map

| Path | Role |
|---|---|
| `space/` | Gradio evaluator app (push to HF Space `main`, not GitHub `main`) |
| `app/channels/twilio_whatsapp.py` | Twilio WhatsApp |
| `app/channels/phone.py` | Africa’s Talking |
| `app/pipeline/space_hotline.py` | Railway → Space `/ask` |
| `app/pipeline/tts.py` | Soro + gTTS |
| `app/pipeline/orchestrator.py` | ASR → intent → RAG → LLM → TTS (gateway/local) |
| `knowledge/` | Curated Nigerian content |
| `docs/architecture.md` | Older internal architecture note |
| `docs/natlas_integration.md` | How weights are loaded |

**Git:** user-facing Railway deploys from branch `1-change-meta-to-twillo-temp`. Hugging Face Space is a **separate** git remote (`spaces/onomeakpobaro/hotline`). Do not `git push origin main` on GitHub to update the Space.
