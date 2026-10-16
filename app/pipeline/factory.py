from __future__ import annotations

import httpx

from app.config import Settings
from app.pipeline.asr import ASRService, NatlasASR
from app.pipeline.llm import LLMService, NatlasLLM


class _SpaceUnused:
    async def transcribe(self, audio_path: str, language: str) -> str:
        raise RuntimeError("natlas_backend=space: N-ATLaS runs on the Hugging Face Space")

    async def generate(self, *args, **kwargs) -> str:
        raise RuntimeError("natlas_backend=space: N-ATLaS runs on the Hugging Face Space")


def build_asr(http_client: httpx.AsyncClient, settings: Settings) -> ASRService:
    if settings.natlas_backend == "space":
        return _SpaceUnused()
    if settings.natlas_backend == "local":
        from app.pipeline.local_natlas import LocalNatlasASR
        return LocalNatlasASR(settings)
    if settings.natlas_backend == "huggingface":
        from app.pipeline.hf_natlas import HuggingFaceASR
        return HuggingFaceASR(http_client, settings)
    return NatlasASR(http_client, settings)


def build_llm(http_client: httpx.AsyncClient, settings: Settings) -> LLMService:
    if settings.natlas_backend == "space":
        return _SpaceUnused()
    if settings.natlas_backend == "local":
        from app.pipeline.local_natlas import LocalNatlasLLM
        return LocalNatlasLLM(settings)
    if settings.natlas_backend == "huggingface":
        from app.pipeline.hf_natlas import HuggingFaceLLM
        return HuggingFaceLLM(http_client, settings)
    return NatlasLLM(http_client, settings)
