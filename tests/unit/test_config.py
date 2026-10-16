from app.config import natlas_origin, natlas_v1_url
from app.utils.phone import is_nigerian_number, normalize_ng_phone


def test_natlas_url_without_v1() -> None:
    assert (
        natlas_v1_url("https://example.com", "audio/transcriptions")
        == "https://example.com/v1/audio/transcriptions"
    )


def test_natlas_url_with_v1() -> None:
    assert natlas_v1_url("https://example.com/v1", "chat/completions") == "https://example.com/v1/chat/completions"


def test_natlas_health_is_on_origin() -> None:
    assert natlas_origin("https://example.com/v1") == "https://example.com"


def test_space_backend_allowed() -> None:
    from app.config import Environment, Settings

    s = Settings(
        natlas_backend="space",
        hf_space_id="onomeakpobaro/hotline",
        at_username="t",
        at_api_key="t",
        at_phone_number="+2348000000000",
        wa_phone_number_id="1",
        wa_access_token="t",
        wa_verify_token="t",
        app_base_url="https://app.example.com",
        api_secret_key="secret",
        phone_hash_salt="test-salt-16chars",
        environment=Environment.DEVELOPMENT,
    )
    assert s.natlas_backend == "space"


def test_base_url_fills_openai_paths() -> None:
    from app.config import Environment, Settings

    s = Settings(
        natlas_backend="gateway",
        natlas_base_url="https://example.com",
        natlas_api_key="k",
        at_username="t",
        at_api_key="t",
        at_phone_number="+2348000000000",
        wa_phone_number_id="1",
        wa_access_token="t",
        wa_verify_token="t",
        app_base_url="https://app.example.com",
        api_secret_key="secret",
        phone_hash_salt="test-salt-16chars",
        environment=Environment.DEVELOPMENT,
    )
    assert s.natlas_asr_url.endswith("/v1/audio/transcriptions")
    assert s.natlas_llm_url.endswith("/v1/chat/completions")


def test_normalize_local_nigerian() -> None:
    assert normalize_ng_phone("08012345678") == "+2348012345678"


def test_normalize_234_without_plus() -> None:
    assert normalize_ng_phone("2348012345678") == "+2348012345678"


def test_is_nigerian() -> None:
    assert is_nigerian_number("08012345678")
    assert not is_nigerian_number("+15555550100")
