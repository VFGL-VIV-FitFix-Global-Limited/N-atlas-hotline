"""Call official NCAIR1 weights via the Hugging Face Inference API.

Same models as https://huggingface.co/NCAIR1/N-ATLaS and the four ASR repos.
Requires HF_TOKEN after accepting the gated model terms on Hugging Face.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from app.config import Settings
from app.models.schemas import Domain
from app.pipeline.asr import ASRError
from app.pipeline.hf_models import (
    ASR_MODEL_IDS,
    LLM_MAX_NEW_TOKENS,
    LLM_MODEL_ID,
    LLM_REPETITION_PENALTY,
    LLM_TEMPERATURE,
)
from app.pipeline.llm import FALLBACK_MESSAGES, GENERAL_PROMPT_TEMPLATE, LANGUAGE_NAMES, RAG_PROMPT_TEMPLATE, SENSITIVE_DOMAINS, SYSTEM_PROMPT
from app.pipeline.natlas_lang import normalise_natlas_language

logger = structlog.get_logger(__name__)


class HuggingFaceASR:
    def __init__(self, config: Settings) -> None:
        self._config = config

    async def transcribe(self, audio_path: str, language: str) -> str:
        try:
            lang = normalise_natlas_language(language)
        except ValueError as e:
            raise ASRError(str(e), status_code=400) from e

        model_id = ASR_MODEL_IDS[lang]
        from huggingface_hub import AsyncInferenceClient

        client = AsyncInferenceClient(token=self._config.hf_token or None, timeout=self._config.asr_timeout_seconds)
        audio = Path(audio_path).read_bytes()
        try:
            result = await client.automatic_speech_recognition(audio, model=model_id)
        except Exception as e:
            logger.error("hf_asr_failed", model=model_id, error=str(e))
            raise ASRError(f"Hugging Face ASR failed ({model_id}): {e}") from e

        text = result.get("text", "") if isinstance(result, dict) else getattr(result, "text", str(result))
        logger.info("hf_asr_success", language=lang, model=model_id, text_length=len(text))
        return text


class HuggingFaceLLM:
    def __init__(self, config: Settings) -> None:
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

        from huggingface_hub import AsyncInferenceClient

        client = AsyncInferenceClient(token=self._config.hf_token or None, timeout=self._config.llm_timeout_seconds)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            result = await client.chat_completion(
                messages=messages,
                model=self._config.natlas_model or LLM_MODEL_ID,
                max_tokens=LLM_MAX_NEW_TOKENS,
                temperature=LLM_TEMPERATURE,
                extra_body={"repetition_penalty": LLM_REPETITION_PENALTY},
            )
        except Exception as e:
            logger.error("hf_llm_failed", model=LLM_MODEL_ID, error=str(e))
            return {
                "en": "I'm sorry, I couldn't generate a response right now. Please try again.",
                "yo": "Ẹ má bínú, mi ò lè dáhùn lọ́wọ́lọ́wọ́. Ẹ jọ̀wọ́ ẹ gbìyànjú lẹ́ẹ̀kan si.",
                "ha": "Yi hakuri, ban iya amsa a yanzu ba. Da fatan za a sake gwadawa.",
                "ig": "Ndo, enweghị m ike ịza ugbu a. Biko nwaa ọzọ.",
            }.get(language, "I'm sorry, I couldn't generate a response right now. Please try again.")

        text = ""
        if hasattr(result, "choices") and result.choices:
            text = result.choices[0].message.content or ""
        elif isinstance(result, dict):
            choices = result.get("choices") or [{}]
            text = (choices[0].get("message") or {}).get("content", "")

        logger.info("hf_llm_success", language=language, domain=domain.value, response_length=len(text))
        return text
