from __future__ import annotations

from typing import Protocol

import httpx
import structlog
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.config import Settings
from app.models.schemas import Domain
from app.pipeline.hf_models import LLM_MAX_NEW_TOKENS, LLM_REPETITION_PENALTY, LLM_TEMPERATURE
from app.pipeline.natlas_lang import normalise_natlas_language

logger = structlog.get_logger(__name__)

SENSITIVE_DOMAINS = {Domain.HEALTH, Domain.GOVERNMENT, Domain.AGRICULTURE}

SYSTEM_PROMPT = """You are N-ATLAS Hotline, a helpful voice assistant for Nigerians.

RULES:
1. Always respond in {language} (the user's selected language).
2. Keep responses under 150 words — this will be spoken aloud.
3. Use simple, conversational language as if speaking to a friend.
4. Only use information from the provided context documents for health, government, agriculture and market queries. Do NOT use your general training knowledge for these sensitive domains.
5. If the context does not contain the answer, say: "I don't have that information right now. You can call [relevant helpline] or visit [relevant office] for help." Then suggest a related topic you CAN help with.
6. For general/casual queries outside sensitive domains (e.g. "what time is it?", "tell me a joke"), you may answer from your general knowledge but preface with: "Based on what I know..."
7. NEVER diagnose medical conditions. For health questions, give general guidance and always advise seeing a doctor for serious symptoms.
8. NEVER give legal advice. Share general rights information and advise consulting a lawyer for specific cases.
9. Include specific details when available: addresses, phone numbers, fees, required documents.
10. For market prices, always state the date the price was recorded.
11. Be warm, respectful, and patient — many users are speaking to AI for the first time.
12. If you are unsure whether your answer is accurate, say so honestly. Trust is more important than sounding confident."""

RAG_PROMPT_TEMPLATE = """CONTEXT INFORMATION:
---
{context}
---

USER QUESTION (transcribed from voice, in {language}):
{question}

Respond in {language}. Use only the context above. Keep it under 150 words. Be conversational — this answer will be spoken aloud to the user."""

GENERAL_PROMPT_TEMPLATE = """USER QUESTION (transcribed from voice, in {language}):
{question}

Respond in {language}. Keep it under 150 words. Be conversational — this answer will be spoken aloud. Preface your answer with "Based on what I know..." since this is outside your curated knowledge base."""

LANGUAGE_NAMES = {"en": "English", "yo": "Yoruba", "ha": "Hausa", "ig": "Igbo"}

FALLBACK_MESSAGES: dict[str, dict[str, str]] = {
    "government": {
        "en": "I don't have specific information on that government service right now. Please call NIMC on 07080606060 for NIN queries, or visit your nearest government office for help.",
        "yo": "Mi ò ní àlàyé nípa iṣẹ́ ìjọba yẹn lọ́wọ́lọ́wọ́. Ẹ jọ̀wọ́ ẹ pe NIMC lórí 07080606060 fún ìbéèrè NIN, tàbí kí ẹ lọ sí ọ́fíìsì ìjọba tó sún mọ́ yín.",
        "ha": "Ba ni bayani kan game da wannan sabis ɗin gwamnati a yanzu. Da fatan za a kira NIMC a 07080606060 don tambayoyin NIN, ko ku ziyarci ofishin gwamnati mafi kusa.",
        "ig": "Enweghị m ozi kpụ ụkpụrụ banyere ọrụ gọọmentị ahụ ugbu a. Biko kpọọ NIMC na 07080606060 maka ajụjụ NIN, ma ọ bụ gaa ọfịs gọọmentị nke dị nso.",
    },
    "health": {
        "en": "I don't have specific information on that health topic. For medical emergencies, please call 112 or visit your nearest hospital. For disease outbreaks, call NCDC on 0800-970-0010.",
        "yo": "Mi ò ní àlàyé nípa ọ̀ràn ìlera yẹn. Fún pàjáwìrì, ẹ pe 112 tàbí lọ sí ilé ìwòsàn tó sún mọ́ yín. Fún àjàkálẹ̀ àrùn, ẹ pe NCDC lórí 0800-970-0010.",
        "ha": "Ba ni bayani kan game da wannan batun lafiya ba. Don gaggawa, a kira 112 ko a ziyarci asibiti mafi kusa. Don annobar cututtuka, a kira NCDC a 0800-970-0010.",
        "ig": "Enweghị m ozi kpụ ụkpụrụ banyere isiokwu ahụ banyere ahụike. Maka mberede, biko kpọọ 112 ma ọ bụ gaa ụlọ ọgwụ nke dị nso. Maka ọrịa na-efe efe, kpọọ NCDC na 0800-970-0010.",
    },
    "agriculture": {
        "en": "I don't have specific information on that agricultural topic right now. Please contact your local Agricultural Development Programme (ADP) office or call the Federal Ministry of Agriculture.",
        "yo": "Mi ò ní àlàyé nípa ọ̀ràn iṣẹ́ àgbẹ̀ yẹn lọ́wọ́lọ́wọ́. Ẹ jọ̀wọ́ ẹ kàn sí ọ́fíìsì ADP àdúgbò yín tàbí Ilé-iṣẹ́ Àpapọ̀ ti Iṣẹ́ Àgbẹ̀.",
        "ha": "Ba ni bayani kan game da wannan batun noma ba a yanzu. Da fatan za a tuntuɓi ofishin ADP na yankinku ko Hukumar Noma ta Tarayya.",
        "ig": "Enweghị m ozi kpụ ụkpụrụ banyere isiokwu ọrụ ugbo ahụ ugbu a. Biko kpọtụrụ ọfịs ADP obodo gị ma ọ bụ kpọọ Ministri Ọrụ Ugbo nke Federal.",
    },
}


class LLMService(Protocol):
    async def generate(
        self, query: str, context: str, language: str, domain: Domain, is_out_of_knowledge: bool
    ) -> str: ...


class NatlasLLM:
    """Concrete LLM implementation hitting N-ATLAS API."""

    def __init__(self, client: httpx.AsyncClient, config: Settings) -> None:
        self._client = client
        self._config = config

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception(
            lambda exc: isinstance(exc, httpx.TimeoutException)
            or (isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500)
        ),
        before_sleep=before_sleep_log(logger, "WARNING"),
    )
    async def generate(
        self,
        query: str,
        context: str,
        language: str,
        domain: Domain,
        is_out_of_knowledge: bool,
    ) -> str:
        """Generate a response using N-ATLAS LLM."""
        # Sensitive domain + out of knowledge → return canned fallback, skip LLM
        if is_out_of_knowledge and domain in SENSITIVE_DOMAINS:
            fallback = FALLBACK_MESSAGES.get(domain.value, {}).get(
                language, FALLBACK_MESSAGES.get(domain.value, {}).get("en", "")
            )
            logger.info("llm_fallback_used", domain=domain.value, language=language)
            return fallback

        lang_name = LANGUAGE_NAMES.get(language, "English")
        system_prompt = SYSTEM_PROMPT.format(language=lang_name)

        if is_out_of_knowledge:
            # General query with no RAG context
            user_prompt = GENERAL_PROMPT_TEMPLATE.format(language=lang_name, question=query)
        else:
            user_prompt = RAG_PROMPT_TEMPLATE.format(
                context=context, language=lang_name, question=query
            )

        try:
            try:
                lang = normalise_natlas_language(language)
            except ValueError:
                lang = "en"

            response = await self._client.post(
                self._config.natlas_llm_url,
                json={
                    "model": self._config.natlas_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": LLM_MAX_NEW_TOKENS,
                    "temperature": LLM_TEMPERATURE,
                    "repetition_penalty": LLM_REPETITION_PENALTY,
                    "language": lang,
                    "stream": False,
                },
                headers={"Authorization": f"Bearer {self._config.natlas_api_key}"},
                timeout=self._config.llm_timeout_seconds,
            )

            request_id = response.headers.get("X-Request-Id")

            if response.status_code == 401:
                logger.error("llm_auth_failed", request_id=request_id)
                return self._get_error_fallback(language)

            if 400 <= response.status_code < 500:
                logger.error(
                    "llm_client_error",
                    status=response.status_code,
                    detail=response.text,
                    request_id=request_id,
                )
                return self._get_error_fallback(language)

            response.raise_for_status()
            data = response.json()
            # Gateway returns vLLM's body unchanged (OpenAI chat shape)
            choices = data.get("choices") or [{}]
            text = (choices[0].get("message") or {}).get("content") or ""

            logger.info(
                "llm_success",
                language=lang,
                domain=domain.value,
                response_length=len(text),
                request_id=request_id,
            )
            return text

        except (httpx.TimeoutException, httpx.HTTPStatusError):
            raise
        except Exception as e:
            logger.error("llm_unexpected_error", error=str(e))
            return self._get_error_fallback(language)

    def _get_error_fallback(self, language: str) -> str:
        messages = {
            "en": "I'm sorry, I couldn't generate a response right now. Please try again.",
            "yo": "Ẹ má bínú, mi ò lè dáhùn lọ́wọ́lọ́wọ́. Ẹ jọ̀wọ́ ẹ gbìyànjú lẹ́ẹ̀kan si.",
            "ha": "Yi hakuri, ban iya amsa a yanzu ba. Da fatan za a sake gwadawa.",
            "ig": "Ndo, enweghị m ike ịza ugbu a. Biko nwaa ọzọ.",
        }
        return messages.get(language, messages["en"])
