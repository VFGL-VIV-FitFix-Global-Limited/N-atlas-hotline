from starlette.requests import Request

from app.channels.twilio_whatsapp import signature_is_valid, twilio_signed_urls, validate_twilio_signature
from app.config import Settings


def _sign(url: str, params: dict, token: str) -> str:
    from base64 import b64encode
    import hashlib
    import hmac

    s = url
    for key in sorted(params.keys()):
        s += key + params[key]
    mac = hmac.new(token.encode("utf-8"), s.encode("utf-8"), hashlib.sha1)
    return b64encode(mac.digest()).decode("utf-8")


def test_validates_public_url_when_request_is_internal(settings: Settings) -> None:
    token = "test-auth-token"
    settings.twilio_auth_token = token
    params = {"Body": "hello", "From": "whatsapp:+2348012345678", "MessageSid": "SM1"}
    public = "https://example.com/twilio/webhook"
    signature = _sign(public, params, token)

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/twilio/webhook",
        "raw_path": b"/twilio/webhook",
        "query_string": b"",
        "headers": [
            (b"host", b"0.0.0.0:8000"),
            (b"x-twilio-signature", signature.encode()),
        ],
        "client": ("100.64.0.1", 1234),
        "server": ("0.0.0.0", 8000),
    }
    request = Request(scope)
    assert public in twilio_signed_urls(request, settings)
    assert signature_is_valid(request, params, settings)


def test_rejects_wrong_signature(settings: Settings) -> None:
    settings.twilio_auth_token = "test-auth-token"
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/twilio/webhook",
        "raw_path": b"/twilio/webhook",
        "query_string": b"",
        "headers": [
            (b"host", b"0.0.0.0:8000"),
            (b"x-twilio-signature", b"not-a-real-signature===="),
        ],
        "client": ("100.64.0.1", 1234),
        "server": ("0.0.0.0", 8000),
    }
    request = Request(scope)
    assert not signature_is_valid(request, {"Body": "hello"}, settings)


def test_validate_direct_match() -> None:
    params = {"Body": "hi"}
    url = "https://hotline.example/twilio/webhook"
    token = "secret"
    sig = _sign(url, params, token)
    assert validate_twilio_signature(url, params, sig, token)
    assert not validate_twilio_signature(url, params, sig, "other")
