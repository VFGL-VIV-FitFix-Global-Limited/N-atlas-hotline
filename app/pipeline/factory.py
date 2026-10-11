from __future__ import annotations

import httpx

from app.config import Settings
from app.pipeline.asr import ASRService, NatlasASR
from app.pipeline.hf_natlas import HuggingFaceASR, HuggingFaceLLM
from app.pipeline.llm import LLMService, NatlasLLM
from app.pipeline.local_natlas import LocalNatlasASR, LocalNatlasLLM


def build_asr(http_client: httpx.AsyncClient, settings: Settings) -> ASRService:
    if settings.natlas_backend == "local":
        return LocalNatlasASR(settings)
    if settings.natlas_backend == "huggingface":
        return HuggingFaceASR(settings)
    return NatlasASR(http_client, settings)


def build_llm(http_client: httpx.AsyncClient, settings: Settings) -> LLMService:
    if settings.natlas_backend == "local":
        return LocalNatlasLLM(settings)
    if settings.natlas_backend == "huggingface":
        return HuggingFaceLLM(settings)
    return NatlasLLM(http_client, settings)
