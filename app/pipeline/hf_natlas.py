"""Call official NCAIR1 weights via the Hugging Face Inference HTTP API.

Bypasses huggingface_hub.AsyncInferenceClient — that client raises
`coroutine raised StopIteration` when a model has no Inference Provider mapping.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import structlog

from app.config import Settings
from app.models.schemas import Domain
from app.pipeline.asr import ASRError
from app.pipeline.hf_models import (
    ASR_MODEL_IDS,
    ASR_SERVERLESS_MODEL,
    LLM_MAX_NEW_TOKENS,
    LLM_MODEL_ID,
    LLM_REPETITION_PENALTY,
    LLM_SERVERLESS_MODEL,
    LLM_TEMPERATURE,
)
from app.pipeline.llm import (
    FALLBACK_MESSAGES,
    GENERAL_PROMPT_TEMPLATE,
    LANGUAGE_NAMES,
    RAG_PROMPT_TEMPLATE,
    SENSITIVE_DOMAINS,
    SYSTEM_PROMPT,
)
from app.pipeline.natlas_lang import normalise_natlas_language
from app.utils.audio import convert_to_wav

logger = structlog.get_logger(__name__)

_HF_INFERENCE_URLS = (
    "https://router.huggingface.co/hf-inference/models/{model}",
    "https://router.huggingface.co/hf-inference/models/{model}/automatic-speech-recognition",
)
_HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"


def _auth_headers(token: str) -> dict[str, str]:
    headers = {"x-wait-for-model": "true"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _extract_asr_text(payload: object) -> str:
    if isinstance(payload, dict):
        text = payload.get("text")
        if isinstance(text, str):
            return text
        inner = payload.get("result")
        if isinstance(inner, dict) and isinstance(inner.get("text"), str):
            return inner["text"]
    if isinstance(payload, list) and payload:
        return _extract_asr_text(payload[0])
    return ""


class HuggingFaceASR:
    def __init__(self, client: httpx.AsyncClient, config: Settings) -> None:
        self._client = client
        self._config = config

    async def transcribe(self, audio_path: str, language: str) -> str:
        try:
            lang = normalise_natlas_language(language)
        except ValueError as e:
            raise ASRError(str(e), status_code=400) from e

        model_id = (
            ASR_MODEL_IDS[lang]
            if self._config.hf_asr_endpoint
            else (self._config.hf_asr_model or ASR_SERVERLESS_MODEL)
        )
        wav_path = audio_path
        try:
            if Path(audio_path).suffix.lower() != ".wav":
                wav_path = await convert_to_wav(audio_path)
            audio = Path(wav_path).read_bytes()
            text = await self._transcribe_bytes(audio, model_id, lang)
        except ASRError:
            raise
        except Exception as e:
            logger.error("hf_asr_failed", model=model_id, error=str(e))
            raise ASRError(f"Hugging Face ASR failed ({model_id}): {e}") from e
        finally:
            if wav_path != audio_path:
                Path(wav_path).unlink(missing_ok=True)

        logger.info("hf_asr_success", language=lang, model=model_id, text_length=len(text))
        return text

    async def _transcribe_bytes(self, audio: bytes, model_id: str, language: str) -> str:
        headers = {
            **_auth_headers(self._config.hf_token),
            "Content-Type": "audio/wav",
        }
        logger.info("hf_asr_request", model=model_id, language=language, bytes=len(audio))
        last_error = "no Hugging Face ASR endpoint responded"
        urls = self._asr_urls(model_id)
        for url in urls:
            try:
                response = await self._client.post(
                    url,
                    content=audio,
                    headers=headers,
                    timeout=self._config.asr_timeout_seconds,
                )
            except httpx.RequestError as e:
                last_error = f"{type(e).__name__} talking to {url}: {e}"
                logger.warning("hf_asr_request_error", url=url, error=str(e))
                continue

            if response.status_code == 503:
                last_error = f"model loading (503) at {url}: {response.text[:300]}"
                logger.warning("hf_asr_model_loading", url=url, body=response.text[:300])
                continue

            if response.status_code >= 400:
                last_error = f"HTTP {response.status_code} from {url}: {response.text[:300]}"
                logger.error("hf_asr_http_error", url=url, status=response.status_code, body=response.text[:300])
                continue

            text = _extract_asr_text(response.json())
            if text.strip():
                return text
            last_error = f"empty transcript from {url}: {response.text[:300]}"

        raise ASRError(f"Hugging Face ASR failed ({model_id}): {last_error}")

    def _asr_urls(self, model_id: str) -> list[str]:
        if self._config.hf_asr_endpoint:
            return [self._config.hf_asr_endpoint.rstrip("/")]
        return [template.format(model=model_id) for template in _HF_INFERENCE_URLS]


class HuggingFaceLLM:
    def __init__(self, client: httpx.AsyncClient, config: Settings) -> None:
        self._client = client
        self._config = config

    async def generate(
        self,
        query: str,
        context: str,
        language: str,
        domain: Domain,
        is_out_of_knowledge: bool,
    ) -> str:
        if is_out_of_knowledge and domain in SENSITIVE_DOMAINS:
            return FALLBACK_MESSAGES.get(domain.value, {}).get(
                language, FALLBACK_MESSAGES.get(domain.value, {}).get("en", "")
            )

        lang_name = LANGUAGE_NAMES.get(language, "English")
        system_prompt = SYSTEM_PROMPT.format(language=lang_name)
        if is_out_of_knowledge:
            user_prompt = GENERAL_PROMPT_TEMPLATE.format(language=lang_name, question=query)
        else:
            user_prompt = RAG_PROMPT_TEMPLATE.format(context=context, language=lang_name, question=query)

        model_id = (
            (self._config.natlas_model or LLM_MODEL_ID)
            if self._config.hf_llm_endpoint
            else (self._config.hf_chat_model or LLM_SERVERLESS_MODEL)
        )
        fallback = {
            "en": "I'm sorry, I couldn't generate a response right now. Please try again.",
            "yo": "Ẹ má bínú, mi ò lè dáhùn lọ́wọ́lọ́wọ́. Ẹ jọ̀wọ́ ẹ gbìyànjú lẹ́ẹ̀kan si.",
            "ha": "Yi hakuri, ban iya amsa a yanzu ba. Da fatan za a sake gwadawa.",
            "ig": "Ndo, enweghị m ike ịza ugbu a. Biko nwaa ọzọ.",
        }.get(language, "I'm sorry, I couldn't generate a response right now. Please try again.")

        try:
            text = await self._chat(model_id, system_prompt, user_prompt)
        except Exception as e:
            logger.error("hf_llm_failed", model=model_id, error=str(e))
            return fallback

        if not text.strip():
            logger.error("hf_llm_empty", model=model_id)
            return fallback

        logger.info("hf_llm_success", language=language, domain=domain.value, response_length=len(text))
        return text

    async def _chat(self, model_id: str, system_prompt: str, user_prompt: str) -> str:
        headers = {
            **_auth_headers(self._config.hf_token),
            "Content-Type": "application/json",
        }
        chat_url = _HF_CHAT_URL
        if self._config.hf_llm_endpoint:
            base = self._config.hf_llm_endpoint.rstrip("/")
            chat_url = base if base.endswith("/chat/completions") else f"{base}/v1/chat/completions"
        response = await self._client.post(
            chat_url,
            headers=headers,
            json={
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": LLM_MAX_NEW_TOKENS,
                "temperature": LLM_TEMPERATURE,
                "repetition_penalty": LLM_REPETITION_PENALTY,
            },
            timeout=self._config.llm_timeout_seconds,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:400]}")
        data = response.json()
        choices = data.get("choices") or [{}]
        return ((choices[0].get("message") or {}).get("content") or "").strip()
