"""Load official N-ATLaS weights locally — Hugging Face model card.

LLM: AutoModelForCausalLM.from_pretrained("NCAIR1/N-ATLaS")
ASR: transformers pipeline on NCAIR1/{Hausa,Igbo,Yoruba,NigerianAccentedEnglish}-ASR
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
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
from app.pipeline.llm import (
    FALLBACK_MESSAGES,
    GENERAL_PROMPT_TEMPLATE,
    LANGUAGE_NAMES,
    RAG_PROMPT_TEMPLATE,
    SENSITIVE_DOMAINS,
    SYSTEM_PROMPT,
)
from app.pipeline.natlas_lang import normalise_natlas_language

logger = structlog.get_logger(__name__)


def _error_text(language: str) -> str:
    return {
        "en": "I'm sorry, I couldn't generate a response right now. Please try again.",
        "yo": "Ẹ má bínú, mi ò lè dáhùn lọ́wọ́lọ́wọ́. Ẹ jọ̀wọ́ ẹ gbìyànjú lẹ́ẹ̀kan si.",
        "ha": "Yi hakuri, ban iya amsa a yanzu ba. Da fatan za a sake gwadawa.",
        "ig": "Ndo, enweghị m ike ịza ugbu a. Biko nwaa ọzọ.",
    }.get(language, "I'm sorry, I couldn't generate a response right now. Please try again.")


class LocalNatlasLLM:
    """In-process NCAIR1/N-ATLaS via transformers (no chat API)."""

    def __init__(self, config: Settings) -> None:
        self._config = config
        self._tokenizer, self._model = self._load()

    def _load(self) -> tuple[object, object]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = self._config.natlas_model or LLM_MODEL_ID
        token = self._config.hf_token or None
        logger.info("loading_natlas_llm", model=model_id, device=self._config.natlas_device)

        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        kwargs: dict = {
            "token": token,
            "device_map": self._config.natlas_device if self._config.natlas_device != "cpu" else None,
        }
        if self._config.natlas_device == "cpu":
            kwargs["torch_dtype"] = torch.float32
        else:
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        model.eval()
        logger.info("natlas_llm_loaded", model=model_id)
        return tokenizer, model

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
            user_prompt = RAG_PROMPT_TEMPLATE.format(
                context=context, language=lang_name, question=query
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, self._generate_sync, messages)
        except Exception:
            logger.exception("local_llm_failed")
            return _error_text(language)

    def _generate_sync(self, messages: list[dict[str, str]]) -> str:
        import torch

        date_string = datetime.now(timezone.utc).strftime("%d %b %Y")
        text = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            date_string=date_string,
        )
        inputs = self._tokenizer(text, return_tensors="pt", add_special_tokens=False)
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=LLM_MAX_NEW_TOKENS,
                use_cache=True,
                repetition_penalty=LLM_REPETITION_PENALTY,
                temperature=LLM_TEMPERATURE,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class LocalNatlasASR:
    """In-process NCAIR1 Whisper checkpoints (one per language)."""

    def __init__(self, config: Settings) -> None:
        self._config = config
        self._pipelines: dict[str, object] = {}

    def _pipeline_for(self, lang: str) -> object:
        if lang in self._pipelines:
            return self._pipelines[lang]

        from transformers import pipeline

        model_id = ASR_MODEL_IDS[lang]
        device = 0 if self._config.natlas_device == "cuda" else -1
        logger.info("loading_natlas_asr", model=model_id)
        asr = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            token=self._config.hf_token or None,
            device=device,
        )
        self._pipelines[lang] = asr
        return asr

    async def transcribe(self, audio_path: str, language: str) -> str:
        try:
            lang = normalise_natlas_language(language)
        except ValueError as e:
            raise ASRError(str(e), status_code=400) from e

        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, self._transcribe_sync, audio_path, lang)
        except ASRError:
            raise
        except Exception as e:
            logger.exception("local_asr_failed", language=lang)
            raise ASRError(f"Local N-ATLaS ASR failed: {e}") from e

    def _transcribe_sync(self, audio_path: str, lang: str) -> str:
        asr = self._pipeline_for(lang)
        result = asr(audio_path)
        text = result["text"] if isinstance(result, dict) else str(result)
        logger.info("local_asr_success", language=lang, text_length=len(text))
        return text
