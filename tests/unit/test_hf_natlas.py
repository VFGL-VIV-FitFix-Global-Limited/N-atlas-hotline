from __future__ import annotations

import httpx
import pytest
import respx

from app.pipeline.factory import build_asr
from app.pipeline.hf_natlas import HuggingFaceASR, _extract_asr_text


def test_extract_asr_text_dict() -> None:
    assert _extract_asr_text({"text": "hello"}) == "hello"


def test_extract_asr_text_list() -> None:
    assert _extract_asr_text([{"text": "hello"}]) == "hello"


def test_huggingface_factory(settings) -> None:
    settings.natlas_backend = "huggingface"
    asr = build_asr(httpx.AsyncClient(), settings)
    assert isinstance(asr, HuggingFaceASR)


@pytest.mark.asyncio
@respx.mock
async def test_transcribe_posts_wav_to_hf(settings, tmp_path) -> None:
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFF....WAVE")
    settings.natlas_backend = "huggingface"
    settings.hf_token = "hf_test"
    route = respx.post(
        "https://router.huggingface.co/hf-inference/models/NCAIR1/NigerianAccentedEnglish"
    ).mock(return_value=httpx.Response(200, json={"text": "how much is rice"}))
    async with httpx.AsyncClient() as client:
        asr = HuggingFaceASR(client, settings)
        text = await asr.transcribe(str(wav), "en")
    assert text == "how much is rice"
    assert route.called
