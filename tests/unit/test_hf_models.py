from app.config import Settings
from app.pipeline.factory import build_asr, build_llm
from app.pipeline.hf_models import ASR_MODEL_IDS, LLM_MODEL_ID
from app.pipeline.local_natlas import LocalNatlasASR


def test_official_llm_repo() -> None:
    assert LLM_MODEL_ID == "NCAIR1/N-ATLaS"


def test_asr_repos_are_ncair1() -> None:
    assert ASR_MODEL_IDS == {
        "ha": "NCAIR1/Hausa-ASR",
        "ig": "NCAIR1/Igbo-ASR",
        "yo": "NCAIR1/Yoruba-ASR",
        "en": "NCAIR1/NigerianAccentedEnglish",
    }


def test_local_backend_uses_from_pretrained_classes(settings: Settings) -> None:
    settings.natlas_backend = "local"
    assert isinstance(build_asr(None, settings), LocalNatlasASR)  # type: ignore[arg-type]
    # Do not construct LocalNatlasLLM here — that downloads the 8B weights.
