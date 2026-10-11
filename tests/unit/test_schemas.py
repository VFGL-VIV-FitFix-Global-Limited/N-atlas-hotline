import pytest
from pydantic import ValidationError

from app.models.schemas import Channel, Domain, Language, PipelineRequest


def test_valid_pipeline_request() -> None:
    req = PipelineRequest(
        session_id="s1",
        channel=Channel.PHONE,
        audio_path="/tmp/a.wav",
        language=Language.ENGLISH,
        phone_number="+2348012345678",
    )
    assert req.language == Language.ENGLISH


def test_invalid_language_rejected() -> None:
    with pytest.raises(ValidationError):
        PipelineRequest(
            session_id="s1",
            channel=Channel.PHONE,
            audio_path="/tmp/a.wav",
            language="fr",  # type: ignore[arg-type]
            phone_number="+2348012345678",
        )


def test_invalid_channel_rejected() -> None:
    with pytest.raises(ValidationError):
        PipelineRequest(
            session_id="s1",
            channel="sms",  # type: ignore[arg-type]
            audio_path="/tmp/a.wav",
            language=Language.YORUBA,
            phone_number="+2348012345678",
        )


def test_invalid_domain_rejected() -> None:
    with pytest.raises(ValueError):
        Domain("legal")


def test_missing_required_fields() -> None:
    with pytest.raises(ValidationError):
        PipelineRequest(session_id="s1")  # type: ignore[call-arg]
